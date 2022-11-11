mod descriptor;
mod filter;

pub use filter::*;

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use crate::prelude::*;

    #[test]
    fn test_create_filter() {
        let data = [[[[0.0, 1.0]]], [[[2.0, 3.0]]]];
        let f = Filter::create(
            CudaDeviceBuilder::new(0)
                .build()
                .unwrap()
                .take(Rc::new(data))
                .unwrap(),
        )
        .unwrap();
        let on_gpu = *f.get_data().unwrap();
        assert_eq!(data, on_gpu);
    }
}
