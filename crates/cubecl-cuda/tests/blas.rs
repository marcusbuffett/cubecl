//! Optional vendor GEMMs must obey the same ordering and failure rules as kernels.
#![cfg(feature = "cublas")]
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cuda::{
    CudaRuntime,
    blas::{Gemm, GemmPrecision},
};
use cubecl_server::runtime::Runtime;

#[cube(launch)]
fn add_one(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

#[test]
#[ignore = "requires CUDA and cuBLAS"]
fn vendor_gemm_preserves_strides_batches_and_kernel_ordering() {
    let client = CudaRuntime::client(&Default::default());
    let (m, n, k) = (2usize, 3usize, 4usize);
    for precision in [GemmPrecision::Tf32, GemmPrecision::Bf16] {
        for batches in [1, 2] {
            for ta in [false, true] {
                for tb in [false, true] {
                    let lda = if ta { m + 1 } else { k + 1 };
                    let ldb = if tb { k + 1 } else { n + 1 };
                    let sa = lda * if ta { k } else { m };
                    let sb = ldb * if tb { n } else { k };
                    let mut a = vec![0.0_f32; batches * sa];
                    let mut b = vec![0.0_f32; batches * sb];
                    let value_a = |batch, row, col| (1 + batch + row + col) as f32 / 8.0;
                    let value_b = |batch, row, col| (1 + 2 * batch + row + col) as f32 / 4.0;
                    for batch in 0..batches {
                        for row in 0..m {
                            for col in 0..k {
                                a[batch * sa
                                    + if ta { col * lda + row } else { row * lda + col }] =
                                    value_a(batch, row, col);
                            }
                        }
                        for row in 0..k {
                            for col in 0..n {
                                b[batch * sb
                                    + if tb { col * ldb + row } else { row * ldb + col }] =
                                    value_b(batch, row, col);
                            }
                        }
                    }
                    let a = client.create_from_slice(f32::as_bytes(&a));
                    let b = client.create_from_slice(f32::as_bytes(&b));
                    let count = batches * m * n;
                    let c = client.empty(count * 4);
                    let out = client.empty(count * 4);
                    let gemm = Gemm {
                        precision,
                        m: m as i32,
                        n: n as i32,
                        k: k as i32,
                        batches: batches as i32,
                        lhs_transposed: ta,
                        rhs_transposed: tb,
                        lhs_leading: lda as i32,
                        rhs_leading: ldb as i32,
                        out_leading: n as i32,
                        lhs_batch_stride: sa as i64,
                        rhs_batch_stride: sb as i64,
                        out_batch_stride: (m * n) as i64,
                    };
                    // SAFETY: descriptors match the allocated padded matrices; outputs are disjoint.
                    unsafe {
                        gemm.launch(&client, a.binding(), b.binding(), c.clone().binding());
                    }
                    add_one::launch(
                        &client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new(&client, 32),
                        unsafe { BufferArg::from_raw_parts(c, count) },
                        unsafe { BufferArg::from_raw_parts(out.clone(), count) },
                    );
                    let data = client.read_one(out).unwrap();
                    let actual = f32::from_bytes(&data);
                    for batch in 0..batches {
                        for row in 0..m {
                            for col in 0..n {
                                let expected = 1.0
                                    + (0..k)
                                        .map(|j| value_a(batch, row, j) * value_b(batch, j, col))
                                        .sum::<f32>();
                                assert!(
                                    (actual[(batch * m + row) * n + col] - expected).abs() < 1e-4
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
