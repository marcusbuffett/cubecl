//! Optional cuBLAS GEMM submitted through `CubeCL`'s stream and failure tracking.
use cubecl_core::{
    client::Client,
    ir::{ElemType, UIntKind},
    server::{BufferBinding, CubeCount, KernelArguments},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::{
    id::KernelId,
    kernel::{BufferIOAttr, CubeKernel, KernelDefinition, KernelMetadata},
    server::{LaunchError, ServerError},
};
use cudarc::cublas::{result, sys};

/// Product precision for a GEMM with FP32 inputs, output and accumulation.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum GemmPrecision {
    /// TF32 tensor-core products (10 fraction bits).
    Tf32,
    /// BF16 tensor-core products (7 fraction bits); explicitly reduced precision.
    Bf16,
    /// FP16 inputs and output with FP32 accumulation (FP16 serving models).
    F16,
}

/// Row-major output `C = A @ B` (FP32, or FP16 with `GemmPrecision::F16`), with FP32 accumulation.
/// Transposed operands use column-major storage; leading dimensions and batch
/// strides are expressed in elements. Buffers are managed by the calling client.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Gemm {
    /// Precision permitted for multiplication; storage and accumulation stay FP32.
    pub precision: GemmPrecision,
    /// Logical output rows.
    pub m: i32,
    /// Logical output columns.
    pub n: i32,
    /// Contracted dimension.
    pub k: i32,
    /// Number of independent matrices.
    pub batches: i32,
    /// Whether A is stored column-major.
    pub lhs_transposed: bool,
    /// Whether B is stored column-major.
    pub rhs_transposed: bool,
    /// Physical leading dimension of A.
    pub lhs_leading: i32,
    /// Physical leading dimension of B.
    pub rhs_leading: i32,
    /// Row stride of C.
    pub out_leading: i32,
    /// Element offset between A matrices.
    pub lhs_batch_stride: i64,
    /// Element offset between B matrices.
    pub rhs_batch_stride: i64,
    /// Element offset between C matrices.
    pub out_batch_stride: i64,
}

pub(crate) struct GemmKernel(pub Gemm);

impl KernelMetadata for GemmKernel {
    fn name(&self) -> &'static str {
        match self.0.precision {
            GemmPrecision::Tf32 => "cublas_tf32_gemm",
            GemmPrecision::Bf16 => "cublas_bf16_gemm",
            GemmPrecision::F16 => "cublas_f16_gemm",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.0)
    }
    fn address_type(&self) -> ElemType {
        ElemType::UInt(UIntKind::U32)
    }
}
impl CubeKernel for GemmKernel {
    fn define(&self) -> KernelDefinition {
        panic!("cuBLAS GEMM must be submitted to a CUDA client")
    }
}

impl Gemm {
    /// Submit a GEMM on the client's current stream without a host synchronization.
    ///
    /// # Safety
    /// The descriptors must describe valid FP32 matrices in these bindings. Output
    /// storage must not overlap either input or another output batch. All bindings
    /// must belong to this CUDA client and remain valid through the queued launch.
    pub unsafe fn launch(
        self,
        client: &Client,
        lhs: BufferBinding,
        rhs: BufferBinding,
        out: BufferBinding,
    ) {
        assert!(client.name().starts_with("cuda-"), "cuBLAS requires CUDA");
        let bindings = KernelArguments::new()
            .with_buffer_io(lhs, BufferIOAttr::ReadOnly)
            .with_buffer_io(rhs, BufferIOAttr::ReadOnly)
            .with_buffer_io(out, BufferIOAttr::WriteOnly);
        client.launch(
            Box::new(GemmKernel(self)),
            CubeCount::Static(1, 1, 1),
            bindings,
        );
    }
}

fn error(reason: impl core::fmt::Debug) -> ServerError {
    LaunchError::Unknown {
        reason: format!("cuBLAS: {reason:?}"),
        backtrace: BackTrace::capture(),
    }
    .into()
}

/// cuBLAS scratch per handle. A handle given its own workspace never
/// allocates during a GEMM, which is what lets GEMMs be recorded into a CUDA
/// graph (an allocation inside a capture window invalidates it).
const WORKSPACE_BYTES: usize = 32 << 20;

/// One handle per `CubeCL` stream, never reset or shared across concurrent streams.
#[derive(Debug)]
pub(crate) struct BlasHandle(sys::cublasHandle_t, cudarc::driver::sys::CUdeviceptr);
impl BlasHandle {
    pub(crate) fn new(stream: cudarc::driver::sys::CUstream) -> Result<Self, ServerError> {
        // SAFETY: the context is current on the server thread creating the handle.
        let workspace = unsafe { cudarc::driver::result::malloc_sync(WORKSPACE_BYTES) }.map_err(error)?;
        let handle = Self(result::create_handle().map_err(error)?, workspace);
        // SAFETY: the new handle, its workspace and the owning CubeCL stream are live.
        unsafe {
            result::set_stream(handle.0, stream as _).map_err(error)?;
            let status = sys::cublasSetWorkspace_v2(handle.0, workspace as _, WORKSPACE_BYTES);
            if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(error(status));
            }
        }
        Ok(handle)
    }
    pub(crate) unsafe fn gemm(&self, g: Gemm, a: u64, b: u64, c: u64) -> Result<(), ServerError> {
        let op = |transpose| {
            if transpose {
                sys::cublasOperation_t::CUBLAS_OP_T
            } else {
                sys::cublasOperation_t::CUBLAS_OP_N
            }
        };
        let (alpha, beta) = (1.0_f32, 0.0_f32);
        let compute = match g.precision {
            GemmPrecision::Tf32 => sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            GemmPrecision::Bf16 => sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            GemmPrecision::F16 => sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        };
        let storage = match g.precision {
            GemmPrecision::F16 => sys::cudaDataType::CUDA_R_16F,
            GemmPrecision::Tf32 | GemmPrecision::Bf16 => sys::cudaDataType::CUDA_R_32F,
        };
        // Row-major C is column-major C^T: swap A/B and M/N. The library reads
        // host alpha/beta during this call; the actual GEMM stays asynchronous.
        // SAFETY: the caller resolved live buffer bindings on this handle's
        // stream and the launch contract requires valid, nonaliasing matrices.
        unsafe {
            if g.batches == 1 {
                return result::gemm_ex(
                    self.0,
                    op(g.rhs_transposed),
                    op(g.lhs_transposed),
                    g.n,
                    g.m,
                    g.k,
                    (&alpha as *const f32).cast(),
                    b as _,
                    storage,
                    g.rhs_leading,
                    a as _,
                    storage,
                    g.lhs_leading,
                    (&beta as *const f32).cast(),
                    c as _,
                    storage,
                    g.out_leading,
                    compute,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
                .map_err(error);
            }
            result::gemm_strided_batched_ex(
                self.0,
                op(g.rhs_transposed),
                op(g.lhs_transposed),
                g.n,
                g.m,
                g.k,
                (&alpha as *const f32).cast(),
                b as _,
                storage,
                g.rhs_leading,
                g.rhs_batch_stride,
                a as _,
                storage,
                g.lhs_leading,
                g.lhs_batch_stride,
                (&beta as *const f32).cast(),
                c as _,
                storage,
                g.out_leading,
                g.out_batch_stride,
                g.batches,
                compute,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
            .map_err(error)
        }
    }
}
impl Drop for BlasHandle {
    fn drop(&mut self) {
        // SAFETY: the handle is owned only here and destroyed once on the server.
        let _ = unsafe { result::destroy_handle(self.0) };
        // SAFETY: allocated in `new` for this handle only, which is now gone.
        let _ = unsafe { cudarc::driver::result::free_sync(self.1) };
    }
}
