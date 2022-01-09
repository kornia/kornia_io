
mod cv {
    use std::path::Path;
    use image::GenericImageView;

    #[derive(Debug)]
    pub struct Tensor<'a> {
        pub shape: Vec<usize>,
        pub data: &'a Vec<u8>,
    }

    impl Tensor {
        pub fn new(shape: Vec<usize>, data: &Vec<u8>) -> Tensor {
            Tensor{shape: shape, data: data}
        }

        pub fn dims(&self) -> usize {
            self.data.len()
        }

        pub fn get(&self, i0: usize, i1: usize, i2: usize, i3: usize) -> u8 {
            let i = i0 * self.shape[1] * self.shape[2] * self.shape[3];
            let j = i1 * self.shape[2] * self.shape[3];
            let k = i2 * self.shape[3];
            self.data[i + j + k + i3]
        }

        pub fn clone(&self) -> Tensor {
            Tensor {shape: self.shape.clone(), data: &self.data.clone() }
        }

        pub fn add(&self, other: Tensor) -> Tensor {
            let mut data: Vec<u8> = self.data.clone();
            for i in 0..data.len() {
                data[i] += other.data[i];
            }
            Tensor { shape: self.shape.clone(), data: &data }
        }

        pub fn mul(&self, other: Tensor) -> Tensor {
            let mut data: Vec<u8> = self.data.clone();
            for i in 0..data.len() {
                data[i] *= other.data[i];
            }
            Tensor { shape: self.shape.clone(), data: &data }
        }

        pub fn subs(&self, other: Tensor) -> Tensor {
            let mut data: Vec<u8> = self.data.clone();
            for i in 0..data.len() {
                data[i] -= other.data[i];
            }
            Tensor { shape: self.shape.clone(), data: &data }
        }

        pub fn div(&self, other: Tensor) -> Tensor {
            let mut data: Vec<u8> = self.data.clone();
            for i in 0..data.len() {
                data[i] /= other.data[i];
            }
            Tensor { shape: self.shape.clone(), data: &data }
        }

        pub fn from_file(file_path: &str) -> Tensor {
            let img: image::DynamicImage = image::open(&Path::new(file_path)).unwrap();
            let new_shape = Vec::from([1, 3, img.height() as usize, img.width() as usize]);
            let new_data: Vec<u8> = img.to_rgb8().to_vec();
            Tensor { shape: new_shape, data: &new_data }
        }

        pub fn print(&self) -> () {
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    for k in 0..self.shape[2] {
                        for l in 0..self.shape[3] {
                            println!("Index: ({}, {}, {}, {}):", i, j, k, l);
                            println!("Val: {:?}", self.get(i, j, k, l));
                            println!("---------");
                        }
                    }
                }
            }
        }
    }

    pub fn cumsum(data: &Vec<usize>) -> usize {
        let mut acc: usize = 0;
        for x in data {
            acc += x;
        }
        return acc
    } 

    pub fn cumprod(data: &Vec<usize>) -> usize {
        let mut acc: usize = 1;
        for x in data {
            acc *= x;
        }
        return acc
    } 

}

fn main() {
    println!("Hello, world!");
    let shape: Vec<usize> = vec![1, 1, 2, 2];
    let data: Vec<u8> = (0 .. cv::cumprod(&shape)).map(|x| x as u8).collect();
    let t1 = cv::Tensor::new(shape, &data);
    println!("{:?}", t1);
    println!("The tensor has {} dimensions.", t1.dims());

    // loop tensor
    println!("Print tensor");
    t1.print();

    // clone a tensor
    let t2 = t1.clone();

    println!("Sum");
    let t3 = t1.add(t2.clone());
    t3.print();

    println!("Mul");
    let t4 = t1.mul(t2.clone());
    t4.print();

    println!("Subs");
    let t5 = t1.subs(t2.clone());
    t5.print();

    println!("Div");
    let t6 = t1.div(t2.clone());
    t6.print();

    // load image from file
    let img = cv::Tensor::from_file("/home/edgar/Downloads/g279.png");
    println!("{:?}", img);

    println!("Goodbye, world!");
}
