# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
A pure Python wrapper for the NCCL library.

The main purpose is to use NCCL combined with CUDA graph, and support
AMEM (Asynchronous Memory) NCCL features including ncclPause, ncclResume,
and ncclSetGroupID for NCCL memory offloading.

This wrapper provides a flexible way to switch between different versions
of NCCL by changing the environment variable `MEGATRON_NCCL_SO_PATH`.
"""

import ctypes
import logging
import os
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)

# === export types and functions from nccl to Python ===
# for the original nccl definition, please check
# https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    """NCCL unique ID structure for communicator initialization."""
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

ncclDataType_t = ctypes.c_int


class ncclDataTypeEnum:
    """NCCL data type enumeration matching nccl.h definitions."""
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        """Convert PyTorch dtype to NCCL data type."""
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


ncclRedOp_t = ctypes.c_int


class ncclRedOpTypeEnum:
    """NCCL reduction operation enumeration."""
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        """Convert PyTorch ReduceOp to NCCL reduction operation."""
        if op == ReduceOp.SUM:
            return cls.ncclSum
        if op == ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == ReduceOp.MAX:
            return cls.ncclMax
        if op == ReduceOp.MIN:
            return cls.ncclMin
        if op == ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    """Dataclass to describe an NCCL function signature."""
    name: str
    restype: Any
    argtypes: List[Any]


def find_nccl_library() -> str:
    """
    Find the NCCL library path.
    
    Priority:
    1. MEGATRON_NCCL_SO_PATH environment variable
    2. torch.cuda.nccl.version() based path
    3. Default system paths
    """
    # Check environment variable first
    env_path = os.environ.get("MEGATRON_NCCL_SO_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Try to find NCCL from PyTorch
    try:
        # Common paths where NCCL might be installed
        possible_paths = [
            "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
            "/usr/local/lib/libnccl.so.2",
            "/usr/lib/libnccl.so.2",
        ]
        
        # Check CUDA_HOME
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        possible_paths.append(os.path.join(cuda_home, "lib64", "libnccl.so.2"))
        possible_paths.append(os.path.join(cuda_home, "lib", "libnccl.so.2"))
        
        # Check LD_LIBRARY_PATH
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        for path in ld_library_path.split(":"):
            if path:
                possible_paths.append(os.path.join(path, "libnccl.so.2"))
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Fallback to just the library name (let the system find it)
        return "libnccl.so.2"
        
    except Exception:
        return "libnccl.so.2"


class NCCLLibrary:
    """
    A pure Python wrapper for the NCCL library.
    
    This class provides Python bindings for NCCL functions using ctypes,
    including support for AMEM NCCL features (ncclPause, ncclResume, ncclSetGroupID).
    
    Example:
        >>> nccl = NCCLLibrary()
        >>> version = nccl.ncclGetVersion()
        >>> print(f"NCCL version: {version}")
        
        # AMEM NCCL memory offloading
        >>> if nccl.enable_amem_nccl:
        ...     nccl.ncclPause()  # Offload NCCL memory
        ...     # ... do other work ...
        ...     nccl.ncclResume()  # Restore NCCL memory
    """
    
    # Standard NCCL functions
    exported_functions = [
        # const char* ncclGetErrorString(ncclResult_t result)
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        # ncclResult_t ncclGetVersion(int *version);
        Function("ncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        # ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
        Function("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        # ncclResult_t ncclCommInitRank(
        #   ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
        Function(
            "ncclCommInitRank",
            ncclResult_t,
            [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int],
        ),
        # ncclResult_t ncclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        Function(
            "ncclAllReduce",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t ncclAllGather(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclComm_t comm,
        #   cudaStream_t stream);
        Function(
            "ncclAllGather",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t ncclReduceScatter(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
        #   cudaStream_t stream);
        Function(
            "ncclReduceScatter",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t ncclSend(
        #   const void* sendbuff, size_t count, ncclDataType_t datatype,
        #   int dest, ncclComm_t comm, cudaStream_t stream);
        Function(
            "ncclSend",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t ncclRecv(
        #   void* recvbuff, size_t count, ncclDataType_t datatype,
        #   int src, ncclComm_t comm, cudaStream_t stream);
        Function(
            "ncclRecv",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t ncclBroadcast(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ncclDataType_t datatype, int root, ncclComm_t comm,
        #   cudaStream_t stream);
        Function(
            "ncclBroadcast",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        # ncclResult_t ncclCommDestroy(ncclComm_t comm);
        Function("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
    ]
    
    # AMEM NCCL functions for memory offloading
    # These functions are only available in AMEM-enabled NCCL builds
    amem_nccl_functions = [
        # ncclResult_t ncclPause(ncclComm_t comm);
        # Pauses NCCL operations and offloads NCCL memory
        # comm parameter can be set to NULL
        Function("ncclPause", ncclResult_t, [ncclComm_t]),
        # ncclResult_t ncclResume(ncclComm_t comm);
        # Resumes NCCL operations and restores NCCL memory
        # comm parameter can be set to NULL
        Function("ncclResume", ncclResult_t, [ncclComm_t]),
        # ncclResult_t ncclSetGroupID(int group_id);
        # Sets the group ID for NCCL operations
        Function("ncclSetGroupID", ncclResult_t, [ctypes.c_int]),
    ]

    # Class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # Class attribute to store the mapping from library path
    # to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        """
        Initialize the NCCL library wrapper.
        
        Args:
            so_file: Path to the NCCL shared library. If None, will attempt
                    to find the library automatically.
        """
        so_file = so_file or find_nccl_library()
        self.enable_amem_nccl = False

        try:
            if so_file not in NCCLLibrary.path_to_library_cache:
                lib = ctypes.CDLL(so_file)
                NCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = NCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load NCCL library from %s. "
                "It is expected if you are not running on NVIDIA/AMD GPUs. "
                "Otherwise, the nccl library might not exist, be corrupted "
                "or it does not support the current platform %s. "
                "If you already have the library, please set the "
                "environment variable MEGATRON_NCCL_SO_PATH "
                "to point to the correct nccl library path.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in NCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            
            # Load standard NCCL functions
            for func in NCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            
            # Try to load AMEM NCCL functions
            amem_available = True
            for func in NCCLLibrary.amem_nccl_functions:
                try:
                    f = getattr(self.lib, func.name)
                    f.restype = func.restype
                    f.argtypes = func.argtypes
                    _funcs[func.name] = f
                except AttributeError:
                    amem_available = False
                    logger.debug(
                        f"AMEM NCCL function {func.name} not found. "
                        "AMEM NCCL features will be disabled."
                    )
                    break
            
            NCCLLibrary.path_to_dict_mapping[so_file] = _funcs
            
            # Store AMEM availability status
            if not hasattr(NCCLLibrary, '_amem_available'):
                NCCLLibrary._amem_available = {}
            NCCLLibrary._amem_available[so_file] = amem_available
        
        self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]
        self.enable_amem_nccl = NCCLLibrary._amem_available.get(so_file, False)
        
        if self.enable_amem_nccl:
            logger.info("AMEM NCCL features enabled (ncclPause, ncclResume, ncclSetGroupID)")
        else:
            logger.info("AMEM NCCL features not available in this NCCL build")

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        """Get the error string for an NCCL result code."""
        return str(self._funcs["ncclGetErrorString"](result).decode("utf-8"))

    def NCCL_CHECK(self, result: ncclResult_t) -> None:
        """Check NCCL result and raise RuntimeError if not success."""
        if result != 0:
            error_str = self.ncclGetErrorString(result)
            raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetVersion(self) -> str:
        """
        Get the NCCL version string.
        
        Returns:
            Version string in format "major.minor.patch"
        """
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["ncclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0") or "0"
        minor = version_str[1:3].lstrip("0") or "0"
        patch = version_str[3:].lstrip("0") or "0"
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        """Generate a unique NCCL ID for communicator initialization."""
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["ncclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def ncclCommInitRank(
        self, world_size: int, unique_id: ncclUniqueId, rank: int
    ) -> ncclComm_t:
        """
        Initialize an NCCL communicator.
        
        Args:
            world_size: Number of processes in the communicator
            unique_id: Unique ID generated by ncclGetUniqueId
            rank: Rank of the current process
            
        Returns:
            NCCL communicator handle
        """
        comm = ncclComm_t()
        self.NCCL_CHECK(
            self._funcs["ncclCommInitRank"](
                ctypes.byref(comm), world_size, unique_id, rank
            )
        )
        return comm

    def ncclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        """Perform an all-reduce operation."""
        self.NCCL_CHECK(
            self._funcs["ncclAllReduce"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ncclReduceScatter(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        """Perform a reduce-scatter operation."""
        self.NCCL_CHECK(
            self._funcs["ncclReduceScatter"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ncclAllGather(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        """Perform an all-gather operation."""
        self.NCCL_CHECK(
            self._funcs["ncclAllGather"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    def ncclSend(
        self,
        sendbuff: buffer_type,
        count: int,
        datatype: int,
        dest: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        """Send data to a destination rank."""
        self.NCCL_CHECK(
            self._funcs["ncclSend"](sendbuff, count, datatype, dest, comm, stream)
        )

    def ncclRecv(
        self,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        src: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        """Receive data from a source rank."""
        self.NCCL_CHECK(
            self._funcs["ncclRecv"](recvbuff, count, datatype, src, comm, stream)
        )

    def ncclBroadcast(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        """Broadcast data from root to all ranks."""
        self.NCCL_CHECK(
            self._funcs["ncclBroadcast"](
                sendbuff, recvbuff, count, datatype, root, comm, stream
            )
        )

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        """
        Destroy an NCCL communicator.
        
        Warning: This is a collective operation and should be called
        by all ranks. Due to Python's garbage collection, it's often
        better not to call it at all in most cases.
        """
        self.NCCL_CHECK(self._funcs["ncclCommDestroy"](comm))

    # === AMEM NCCL functions for memory offloading ===
    
    def ncclPause(self, comm: Optional[ncclComm_t] = None) -> None:
        """
        Pause NCCL operations and offload NCCL memory.
        
        This function is used to temporarily free NCCL's internal memory
        when it's not needed, allowing other operations to use the GPU memory.
        
        Args:
            comm: NCCL communicator handle. Can be None/NULL.
            
        Raises:
            RuntimeError: If AMEM NCCL is not enabled or the operation fails.
        """
        if not self.enable_amem_nccl:
            raise RuntimeError(
                "AMEM NCCL is not enabled. ncclPause is not available in this NCCL build."
            )
        self.NCCL_CHECK(self._funcs["ncclPause"](comm))
        logger.debug("NCCL paused - memory offloaded")

    def ncclResume(self, comm: Optional[ncclComm_t] = None) -> None:
        """
        Resume NCCL operations and restore NCCL memory.
        
        This function restores NCCL's internal memory after a previous
        ncclPause call, allowing NCCL operations to proceed.
        
        Args:
            comm: NCCL communicator handle. Can be None/NULL.
            
        Raises:
            RuntimeError: If AMEM NCCL is not enabled or the operation fails.
        """
        if not self.enable_amem_nccl:
            raise RuntimeError(
                "AMEM NCCL is not enabled. ncclResume is not available in this NCCL build."
            )
        self.NCCL_CHECK(self._funcs["ncclResume"](comm))
        logger.debug("NCCL resumed - memory restored")

    def ncclSetGroupID(self, group_id: int) -> None:
        """
        Set the group ID for NCCL operations.
        
        Args:
            group_id: The group ID to set.
            
        Raises:
            RuntimeError: If AMEM NCCL is not enabled or the operation fails.
        """
        if not self.enable_amem_nccl:
            raise RuntimeError(
                "AMEM NCCL is not enabled. ncclSetGroupID is not available in this NCCL build."
            )
        self.NCCL_CHECK(self._funcs["ncclSetGroupID"](group_id))
        logger.debug(f"NCCL group ID set to {group_id}")


__all__ = [
    "NCCLLibrary",
    "ncclDataTypeEnum",
    "ncclRedOpTypeEnum",
    "ncclUniqueId",
    "ncclComm_t",
    "cudaStream_t",
    "buffer_type",
    "find_nccl_library",
]

