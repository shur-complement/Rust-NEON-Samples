// Example - dot product
// Original in C++ version at https://learn.arm.com/learning-paths/smartphones-and-mobile/android_neon/dot_product_neon/

use std::arch::aarch64::*;
use std::time::Instant;

fn generate_ramp(start: i16, len: usize) -> Vec<i16> {
    let mut ramp = vec![0; len];
    for i in 0..len {
        ramp[i] = start + (i as i16);
    }
    ramp
}

pub fn dot_product(v1: &[i16], v2: &[i16]) -> i32 {
    debug_assert!(v1.len() == v2.len());
    v1.iter().zip(v2.iter())
      .map(|v| (*v.0 as i32) * (*v.1 as i32))
      .sum()
}

pub fn dot_product_neon(v1: &[i16], v2: &[i16]) -> i32 {
    debug_assert!(v1.len() == v2.len());
    let len = v1.len();
    const TRANSFER_SIZE: usize = 4;
    let segments = len / TRANSFER_SIZE;

unsafe {
    // 4-element vectors of zeros
    let mut sum1 = vdupq_n_s32(0);
    let mut sum2 = vdupq_n_s32(0);
    let mut sum3 = vdupq_n_s32(0);
    let mut sum4 = vdupq_n_s32(0);

    // Main loop (note that loop index goes through segments). Strided by 4.
    let mut v1_ptr = v1.as_ptr();
    let mut v2_ptr = v2.as_ptr();
    for _ in (0..segments).step_by(TRANSFER_SIZE) {
        // Load vector elements to registers
        let v11 = vld1q_s16(v1_ptr);
        let v11_low = vget_low_s16(v11);
        let v11_high = vget_high_s16(v11);

        let v12 = vld1q_s16(v2_ptr);
        let v12_low = vget_low_s16(v12);
        let v12_high = vget_high_s16(v12);

        let v21 = vld1q_s16(v1_ptr.offset(8));
        let v21_low = vget_low_s16(v21);
        let v21_high = vget_high_s16(v21);

        let v22 = vld1q_s16(v2_ptr.offset(8));
        let v22_low = vget_low_s16(v22);
        let v22_high = vget_high_s16(v22);

        // Multiply and accumulate: partial_sums_neon += v1_neon * v2_neon
        sum1 = vmlal_s16(sum1, v11_low, v12_low);
        sum2 = vmlal_s16(sum2, v11_high, v12_high);
        sum3 = vmlal_s16(sum3, v21_low, v22_low);
        sum4 = vmlal_s16(sum4, v21_high, v22_high);

        v1_ptr = v1_ptr.offset(16);
        v2_ptr = v2_ptr.offset(16);
    }

    let mut partial_sums_neon;
    partial_sums_neon = vaddq_s32(sum1, sum2);
    partial_sums_neon = vaddq_s32(partial_sums_neon, sum3);
    partial_sums_neon = vaddq_s32(partial_sums_neon, sum4);

    // Sum up remaining parts
    let remain = len % TRANSFER_SIZE;
    for _ in 0..remain {
        let v1_neon = vld1_s16(v1_ptr);
        let v2_neon = vld1_s16(v2_ptr);
        partial_sums_neon = vmlal_s16(partial_sums_neon, v1_neon, v2_neon);
        v1_ptr = v1_ptr.offset(4);
        v2_ptr = v2_ptr.offset(4);
    }

    // Store partial sums
    let mut partial_sums = [0i32; TRANSFER_SIZE];
    vst1q_s32(partial_sums.as_mut_ptr(), partial_sums_neon);

    // Sum up partial sums
    let mut result = 0i32;
    for i in 0..TRANSFER_SIZE {
        result += partial_sums[i];
    }

    return result;
}
}

fn main() {
    // Ramp length and number of trials
    let ramp_length = 1024;
    let trials = 10000;

    // Generate two input vectors
    // (0, 1, ..., rampLength - 1)
    // (100, 101, ..., 100 + rampLength-1)
    let ramp1 = generate_ramp(0, ramp_length);
    let ramp2 = generate_ramp(100, ramp_length);

    // Without NEON intrinsics
    // Invoke dot_product and measure performance
    let mut last_result = 0;

    let mut start = Instant::now();
    for i in 0..trials {
        last_result = dot_product(&ramp1, &ramp2);
    }
    let elapsed_time = start.elapsed().as_millis();

    // With NEON intrinsics
    // Invoke dotProductNeon and measure performance
    let mut last_result_neon = 0;

    start = Instant::now();
    for _ in 0..trials {
        last_result_neon = dot_product_neon(&ramp1, &ramp2);
    }
    let elapsed_time_neon = start.elapsed().as_millis();

    println!("----==== NO NEON ====----\nResult: {}", last_result);
    println!("\nElapsed time: {} ms", elapsed_time);
    println!("\n\n----==== NEON ====----\n");
    println!("Result: {}", last_result_neon);
    println!("\nElapsed time: {} ms", elapsed_time_neon);
}
