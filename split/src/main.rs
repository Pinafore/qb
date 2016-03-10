extern crate regex;
use regex::Regex;
use std::io;
use std::io::prelude::*;
use std::fs::OpenOptions;
use std::env;

fn main() {
    let args: Vec<_> = env::args().collect();
    let mut feat_file = OpenOptions::new().create(true).write(true).append(true).open(&args[1]).unwrap();
    let mut meta_file = OpenOptions::new().create(true).write(true).append(true).open(&args[2]).unwrap();
    let re = Regex::new(r"\|\|\|").unwrap();
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let string_result = line.unwrap();
        let input = string_result.as_str();
        let splits: Vec<&str> = re.split(input).collect();
        if splits.len() == 0 {
            break;
        }
        feat_file.write(splits[0].trim().as_bytes());
        feat_file.write(b"\n");
        meta_file.write(splits[1].trim().as_bytes());
        meta_file.write(b"\n");
    }
}
