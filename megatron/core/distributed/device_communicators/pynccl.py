# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
PyNCCL Communicator for Megatron.

This module provides a Python wrapper around NCCL communicators that can be
used alongside PyTorch's distributed process groups. It supports AMEM NCCL
features for memory offloading (ncclPause, ncclResume, ncclSetGroupID).

Example usage:
    # Initialize during Megatron setup
    from megatron.core.parallel_state import get_tensor_model_parallel_group
    
    tp_group = get_tensor_model_parallel_group()
    pynccl_comm = tp_group.pynccl_comm
    
    # Offload NCCL memory when not needed
    if pynccl_comm.nccl.enable_amem_nccl:
        pynccl_comm.nccl_pause()
    
    # Restore NCCL memory before communication
    if pynccl_comm.nccl.enable_amem_nccl:
        pynccl_comm.nccl_resume()
"""

import logging
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from .pynccl_wrapper import (
    NCCLLibrary,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
    cudaStream_t,
    buffer_type,
)

logger = logging.getLogger(__name__)


class PyNCCLCommunicator:
    """
    A PyNCCL communicator wrapper for Megatron.
    
    This class wraps the NCCLLibrary to provide a higher-level interface
    for NCCL operations, including support for AMEM NCCL memory offloading.
    
    Attributes:
        nccl: The underlying NCCLLibrary instance
        comm: The NCCL communicator handle
        rank: The rank of this process in the communicator
        world_size: The total number of processes in the communicator
        device: The CUDA device associated with this communicator
        stream: The CUDA stream used for NCCL operations
        is_paused: Whether NCCL is currently paused (memory offloaded)
    """
    
    def __init__(
        self,
        group: Optional[ProcessGroup] = None,
        ranks: Optional[List[int]] = None,
        local_rank: Optional[int] = None,
        device: Optional[Union[int, torch.device]] = None,
        nccl_so_path: Optional[str] = None,
    ):
        """
        Initialize a PyNCCL communicator.
        
        Args:
            group: PyTorch distributed process group. If provided, ranks and
                   local_rank will be inferred from the group.
            ranks: List of global ranks in this communicator. Required if
                   group is not provided.
            local_rank: Local rank within this communicator. Required if
                        group is not provided.
            device: CUDA device to use. Defaults to current device.
            nccl_so_path: Path to NCCL shared library. If None, will be
                         found automatically.
        """
        # Initialize NCCL library
        self.nccl = NCCLLibrary(nccl_so_path)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            self.device = torch.device("cuda", device)
        else:
            self.device = device
        
        # Get rank information from group or parameters
        if group is not None:
            self.world_size = group.size()
            self.group = group
            
            # Use group.rank() method - this is the Megatron standard way
            # and is more reliable than dist.get_rank(group)
            try:
                self.rank = group.rank()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get rank from group {group}: {e}"
                )
            
            self._ranks = list(range(self.world_size))  # Relative ranks
        else:
            if ranks is None or local_rank is None:
                raise ValueError(
                    "Either 'group' or both 'ranks' and 'local_rank' must be provided"
                )
            self.world_size = len(ranks)
            self.rank = local_rank
            self.group = None
            self._ranks = ranks
        
        # Initialize NCCL communicator
        self.comm: Optional[ncclComm_t] = None
        self._initialized = False
        self.is_paused = False
        
        # Create a dedicated stream for NCCL operations
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream()
        
        logger.info(
            f"PyNCCLCommunicator created: rank={self.rank}, "
            f"world_size={self.world_size}, device={self.device}, "
            f"amem_nccl_enabled={self.nccl.enable_amem_nccl}"
        )
    
    def initialize(self, unique_id: Optional[ncclUniqueId] = None) -> None:
        """
        Initialize the NCCL communicator.
        
        This must be called before any communication operations.
        In a distributed setting, the unique_id should be broadcast
        from rank 0 to all other ranks.
        
        Args:
            unique_id: NCCL unique ID for communicator initialization.
                      If None and this is rank 0, a new ID will be generated.
        """
        if self._initialized:
            logger.warning("PyNCCLCommunicator is already initialized")
            return
        
        with torch.cuda.device(self.device):
            # Generate or receive unique ID
            if unique_id is None:
                if self.rank == 0:
                    unique_id = self.nccl.ncclGetUniqueId()
                else:
                    unique_id = ncclUniqueId()
            
            # Broadcast unique ID using PyTorch distributed if group is available
            if self.group is not None and self.world_size > 1:
                # Convert unique_id to tensor for broadcasting
                id_tensor = torch.frombuffer(
                    bytearray(unique_id.internal), dtype=torch.uint8
                ).clone().cuda(self.device)
                
                dist.broadcast(id_tensor, src=0, group=self.group)
                
                # Copy back to unique_id
                unique_id.internal[:] = id_tensor.cpu().numpy().astype("int8")
            
            # Initialize NCCL communicator
            self.comm = self.nccl.ncclCommInitRank(
                self.world_size, unique_id, self.rank
            )
            self._initialized = True
            
        logger.info(f"PyNCCLCommunicator initialized: rank={self.rank}")
    
    def _get_stream_ptr(self, stream: Optional[torch.cuda.Stream] = None) -> cudaStream_t:
        """Get the CUDA stream pointer for NCCL operations."""
        if stream is None:
            stream = self.stream
        return cudaStream_t(stream.cuda_stream)
    
    def _get_buffer_ptr(self, tensor: torch.Tensor) -> buffer_type:
        """Get the buffer pointer for a tensor."""
        return buffer_type(tensor.data_ptr())
    
    # === AMEM NCCL Memory Management ===
    
    def nccl_pause(self) -> None:
        """
        Pause NCCL and offload its internal memory.
        
        This should be called when NCCL communication is not needed
        for an extended period, to free up GPU memory for other operations.
        
        Raises:
            RuntimeError: If AMEM NCCL is not enabled.
        """
        if not self.nccl.enable_amem_nccl:
            raise RuntimeError(
                "AMEM NCCL is not enabled. Cannot pause NCCL."
            )
        
        if self.is_paused:
            logger.warning("NCCL is already paused")
            return
        
        # Synchronize all streams before pausing
        torch.cuda.synchronize(self.device)
        
        # Pass None (NULL) as comm parameter as specified in the requirements
        self.nccl.ncclPause(None)
        self.is_paused = True
        
        logger.info(f"NCCL paused on rank {self.rank} - memory offloaded")
    
    def nccl_resume(self) -> None:
        """
        Resume NCCL and restore its internal memory.
        
        This must be called before any NCCL communication operations
        after a previous nccl_pause call.
        
        Raises:
            RuntimeError: If AMEM NCCL is not enabled.
        """
        if not self.nccl.enable_amem_nccl:
            raise RuntimeError(
                "AMEM NCCL is not enabled. Cannot resume NCCL."
            )
        
        if not self.is_paused:
            logger.warning("NCCL is not paused")
            return
        
        # Pass None (NULL) as comm parameter as specified in the requirements
        self.nccl.ncclResume(None)
        self.is_paused = False
        
        logger.info(f"NCCL resumed on rank {self.rank} - memory restored")
    
    def set_group_id(self, group_id: int) -> None:
        """
        Set the group ID for NCCL operations.
        
        Args:
            group_id: The group ID to set.
            
        Raises:
            RuntimeError: If AMEM NCCL is not enabled.
        """
        if not self.nccl.enable_amem_nccl:
            raise RuntimeError(
                "AMEM NCCL is not enabled. Cannot set group ID."
            )
        
        self.nccl.ncclSetGroupID(group_id)
        logger.debug(f"NCCL group ID set to {group_id} on rank {self.rank}")
    
    @contextmanager
    def pause_context(self):
        """
        Context manager for temporarily pausing NCCL.
        
        Example:
            with pynccl_comm.pause_context():
                # NCCL memory is offloaded here
                do_other_work()
            # NCCL memory is restored here
        """
        if not self.nccl.enable_amem_nccl:
            # If AMEM NCCL is not enabled, just yield without doing anything
            yield
            return
        
        was_paused = self.is_paused
        if not was_paused:
            self.nccl_pause()
        try:
            yield
        finally:
            if not was_paused:
                self.nccl_resume()
    
    # === Communication Operations ===
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Perform an in-place all-reduce operation.
        
        Args:
            tensor: Tensor to reduce (modified in-place)
            op: Reduction operation (SUM, PRODUCT, MAX, MIN, AVG)
            stream: CUDA stream to use. Defaults to the communicator's stream.
        """
        if not self._initialized:
            raise RuntimeError("PyNCCLCommunicator is not initialized")
        
        if self.is_paused:
            raise RuntimeError("NCCL is paused. Call nccl_resume() first.")
        
        self.nccl.ncclAllReduce(
            self._get_buffer_ptr(tensor),
            self._get_buffer_ptr(tensor),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            self._get_stream_ptr(stream),
        )
    
    def all_gather(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Perform an all-gather operation.
        
        Args:
            output_tensor: Output tensor (size = world_size * input_tensor.numel())
            input_tensor: Input tensor to gather
            stream: CUDA stream to use. Defaults to the communicator's stream.
        """
        if not self._initialized:
            raise RuntimeError("PyNCCLCommunicator is not initialized")
        
        if self.is_paused:
            raise RuntimeError("NCCL is paused. Call nccl_resume() first.")
        
        self.nccl.ncclAllGather(
            self._get_buffer_ptr(input_tensor),
            self._get_buffer_ptr(output_tensor),
            input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            self._get_stream_ptr(stream),
        )
    
    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Perform a reduce-scatter operation.
        
        Args:
            output_tensor: Output tensor (size = input_tensor.numel() / world_size)
            input_tensor: Input tensor to reduce and scatter
            op: Reduction operation
            stream: CUDA stream to use. Defaults to the communicator's stream.
        """
        if not self._initialized:
            raise RuntimeError("PyNCCLCommunicator is not initialized")
        
        if self.is_paused:
            raise RuntimeError("NCCL is paused. Call nccl_resume() first.")
        
        self.nccl.ncclReduceScatter(
            self._get_buffer_ptr(input_tensor),
            self._get_buffer_ptr(output_tensor),
            output_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            self._get_stream_ptr(stream),
        )
    
    def broadcast(
        self,
        tensor: torch.Tensor,
        root: int = 0,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Broadcast a tensor from root to all ranks.
        
        Args:
            tensor: Tensor to broadcast (modified in-place on non-root ranks)
            root: Source rank for the broadcast
            stream: CUDA stream to use. Defaults to the communicator's stream.
        """
        if not self._initialized:
            raise RuntimeError("PyNCCLCommunicator is not initialized")
        
        if self.is_paused:
            raise RuntimeError("NCCL is paused. Call nccl_resume() first.")
        
        self.nccl.ncclBroadcast(
            self._get_buffer_ptr(tensor),
            self._get_buffer_ptr(tensor),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            root,
            self.comm,
            self._get_stream_ptr(stream),
        )
    
    def send(
        self,
        tensor: torch.Tensor,
        dest: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Send a tensor to a destination rank.
        
        Args:
            tensor: Tensor to send
            dest: Destination rank
            stream: CUDA stream to use. Defaults to the communicator's stream.
        """
        if not self._initialized:
            raise RuntimeError("PyNCCLCommunicator is not initialized")
        
        if self.is_paused:
            raise RuntimeError("NCCL is paused. Call nccl_resume() first.")
        
        self.nccl.ncclSend(
            self._get_buffer_ptr(tensor),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            dest,
            self.comm,
            self._get_stream_ptr(stream),
        )
    
    def recv(
        self,
        tensor: torch.Tensor,
        src: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        Receive a tensor from a source rank.
        
        Args:
            tensor: Tensor to receive into
            src: Source rank
            stream: CUDA stream to use. Defaults to the communicator's stream.
        """
        if not self._initialized:
            raise RuntimeError("PyNCCLCommunicator is not initialized")
        
        if self.is_paused:
            raise RuntimeError("NCCL is paused. Call nccl_resume() first.")
        
        self.nccl.ncclRecv(
            self._get_buffer_ptr(tensor),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            self._get_stream_ptr(stream),
        )
    
    def synchronize(self) -> None:
        """Synchronize the NCCL stream."""
        self.stream.synchronize()
    
    def destroy(self) -> None:
        """
        Destroy the NCCL communicator.
        
        Warning: This is a collective operation and should be called
        by all ranks. Due to Python's garbage collection, it's often
        better to let the communicator be cleaned up automatically.
        """
        if self._initialized and self.comm is not None:
            # Resume if paused before destroying
            if self.is_paused:
                try:
                    self.nccl_resume()
                except Exception:
                    pass
            
            try:
                self.nccl.ncclCommDestroy(self.comm)
            except Exception as e:
                logger.warning(f"Error destroying NCCL communicator: {e}")
            
            self.comm = None
            self._initialized = False
    
    def __del__(self):
        """Cleanup on deletion - note that we don't call destroy to avoid hangs."""
        pass


__all__ = ["PyNCCLCommunicator"]



