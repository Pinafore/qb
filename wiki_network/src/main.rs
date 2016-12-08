extern crate wiki_network;

use std::io::prelude::*;
use std::fs::File;
use std::io::BufReader;

use wiki_network::graph;

fn main() {
    println!("Hello, world!");
    let f = File::open("/home/pedro/wiki/titles-sorted.txt").expect("titles-sorted.txt doesn't exist");
    let reader = BufReader::new(&f);
    let mut i = 0;
    for raw_line in reader.lines() {
        i += 1;
    }
    println!("Found {} lines", i);
}
