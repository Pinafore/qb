extern crate wiki_network;

use std::io::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::collections::HashSet;

use wiki_network::wikipedia::{WikiGraph, load_questions, load_stopwords, tokenize};

fn main() {
    let stop_words = load_stopwords();
    let questions = load_questions();
    let wiki_graph = WikiGraph::new();
    let ref page_map = wiki_graph.page_map;
    let ref new_vertex_map = wiki_graph.new_vertex_map;
    let ref neighbors = wiki_graph.adjacency_list;
    let ref vertex_page_map = wiki_graph.vertex_page_map;
    //for (i, q) in questions.iter().enumerate() {
    //    println!("Starting iteration {}", i);
    //    let tokens = tokenize(q, &stop_words);
    /*    let mut seeds: HashSet<usize> = HashSet::new();*/
        //let mut candidates: HashSet<usize> = HashSet::new();
        //for word in tokens {
            //if page_map.contains_key(&word) {
                //let v_index = page_map[&word];
                //let new_v_index = new_vertex_map[&v_index];
                //seeds.insert(new_v_index);
                //for node in &neighbors[new_v_index] {
                    ////
                //}
            //}
        //}
    /*}*/
    let albert = "albert_einstein";
    let v_albert = page_map[albert];
    let new_v_albert = new_vertex_map[&v_albert];
    let ref albert_neighbors = neighbors[new_v_albert];
    println!("Num: {}", albert_neighbors.len());
    for n in albert_neighbors {
        println!("Neighbor: {}", vertex_page_map[n]);
    }
}

