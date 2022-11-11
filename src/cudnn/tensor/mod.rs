mod data_type;
mod descriptor;
mod operation;
mod tensor4d;

pub use data_type::*;
pub use operation::*;
pub use tensor4d::*;

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_create_tensor() {
        let data = [[[[0.0, 1.0]]], [[[2.0, 3.0]]]];
        let t = Tensor4D::alloc_with(&CudaDeviceBuilder::new(0).build().unwrap(), data).unwrap();
        let on_gpu = *t.get_data().unwrap();
        assert_eq!(data, on_gpu);
    }
}
