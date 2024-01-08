// Example - Matrix Multiplication
// Original in C at https://developer.arm.com/documentation/102467/0201/Example---matrix-multiplication
// Original in ASM at https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/coding-for-neon---part-3-matrix-multiplication

use std::arch::aarch64::*;

pub fn matmul(A: &[f32; 16], B: &[f32; 16], C: &mut [f32; 16]) {
unsafe {
    let A_ptr = A.as_ptr();
    let B_ptr = B.as_ptr();
    let C_ptr = C.as_mut_ptr();

    let A0 = vld1q_f32(A_ptr);
    let A1 = vld1q_f32(A_ptr.offset(4));
    let A2 = vld1q_f32(A_ptr.offset(8));
    let A3 = vld1q_f32(A_ptr.offset(12));

    let B0 = vld1q_f32(B_ptr);
    let B1 = vld1q_f32(B_ptr.offset(4));
    let B2 = vld1q_f32(B_ptr.offset(8));
    let B3 = vld1q_f32(B_ptr.offset(12));

    // Multiply accumulate in 4x1 blocks, i.e. each column in C
    let mut C0 = vmulq_laneq_f32(A0, B0, 0);
    let mut C1 = vmulq_laneq_f32(A0, B1, 0);
    let mut C2 = vmulq_laneq_f32(A0, B2, 0);
    let mut C3 = vmulq_laneq_f32(A0, B3, 0);

    C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
    C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
    C0 = vfmaq_laneq_f32(C0, A3, B0, 3);
    vst1q_f32(C_ptr, C0); // store

    C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
    C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
    C1 = vfmaq_laneq_f32(C1, A3, B1, 3);
    vst1q_f32(C_ptr.offset(4), C1);

    C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
    C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
    C2 = vfmaq_laneq_f32(C2, A3, B2, 3);
    vst1q_f32(C_ptr.offset(8), C2);

    C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
    C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
    C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
    vst1q_f32(C_ptr.offset(12), C3);
}
}

fn main() {
    let A = [1.,2.,3.,4.,
             5.,6.,5.,8.,
             9.,8.,7.,6.,
             5.,4.,3.,2.];

    let B = [1.,2.,3.,4.,
             5.,6.,5.,8.,
             9.,8.,7.,6.,
             5.,4.,3.,2.];

    let mut C = [0.; 16];

    matmul(&A, &B, &mut C);

    println!("RESULT: {:?}", C);
}
