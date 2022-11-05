//! This example calculates the sum of matrix-vector and
//! transposed-matrix-vector multiplication

use cudarc::prelude::*;

const VECTOR: [f64; 3] = [11.0, 13.0, 17.0];
const MATRIX: [[f64; 3]; 3] = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]];

fn main() -> CublasResult<()> {
    println!("Vector v:");
    println!("( {} )", VECTOR.map(|v| format!("{v:>5.1}")).join(" "));
    println!();
    println!("Matrix m:");
    for row in MATRIX {
        println!("( {} )", row.map(|v| format!("{v:>5.1}")).join(" "));
    }
    println!();
    println!();

    let cublas_handle = CublasHandle::create()?;

    let device = CudaDeviceBuilder::new(0).build().unwrap();

    let vector_allocation = unsafe { device.alloc() }.unwrap();
    let vector = CublasVector::new(vector_allocation, &VECTOR)?;

    let matrix_allocation = unsafe { device.alloc() }.unwrap();
    let mut transposed = [[0.0; 3]; 3];
    for (y, row) in MATRIX.into_iter().enumerate() {
        for (x, cell) in row.into_iter().enumerate() {
            transposed[x][y] = cell;
        }
    }
    let matrix = CublasMatrix::new(matrix_allocation, &transposed)?;

    let out_allocation = unsafe { device.alloc() }.unwrap();
    let mut result = unsafe { CublasVector::uninit(out_allocation) };
    result.gemv(&cublas_handle, &matrix, &vector, false)?;

    let mut result_host = Default::default();
    result.get(&mut result_host)?;

    println!("v * M ");
    println!(
        "= ( {} )",
        result_host.map(|v| format!("{v:>5.1}")).join(" ")
    );
    println!();
    println!();

    result.gemv(&cublas_handle, &matrix.transposed(), &vector, true)?;

    result.get(&mut result_host)?;


    println!("v * M + v * M^T");
    println!(
        "= ( {} )",
        result_host.map(|v| format!("{v:>5.1}")).join(" ")
    );

    Ok(())
}
