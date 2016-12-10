use std::io::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::collections::{HashMap, HashSet};

use regex::Regex;


pub struct WikiGraph {
    pub adjacency_list: Vec<Vec<usize>>,
    rev_adjacency_list: Vec<Vec<usize>>,
    pub page_map: HashMap<String, usize>,
    vertex_set: HashSet<usize>,
    pub new_vertex_map: HashMap<usize, usize>,
    pub vertex_page_map: HashMap<usize, String>
}

pub fn load_stopwords() -> HashSet<String> {
    let f = File::open("data/stopwords.txt").unwrap();
    let reader = BufReader::new(&f);
    let mut stop_words = HashSet::new();
    for raw_line in reader.lines() {
        let w = raw_line.unwrap();
        stop_words.insert(w);
    }
    stop_words
}

pub fn tokenize<'a>(text: &'a str, stop_words: &HashSet<String>) -> Vec<String> {
    text.to_lowercase().replace(',', "").replace('.', "").replace('"', "")
        .split_whitespace()
        .filter(|t| !stop_words.contains(&t.to_string()))
        .map(|t| t.to_string())
        .collect()
}

pub fn load_questions() -> Vec<String> {
    let f = File::open("data/qb_questions.txt").unwrap();
    let reader = BufReader::new(&f);
    let mut questions = Vec::new();
    for raw_line in reader.lines() {
        let q = raw_line.unwrap();
        questions.push(q);
    }
    questions
}

impl WikiGraph {
    pub fn new() -> WikiGraph {
        println!("Running...");
        let stop_words = load_stopwords();

        let titles_file = File::open("data/titles-sorted.txt").unwrap();
        let titles_reader = BufReader::new(&titles_file);
        let page_regex = Regex::new(r"^[a-zA-Z0-9_\(\)']+$").unwrap();
        let mut page_map: HashMap<String, usize> = HashMap::new();
        let mut vertex_set: HashSet<usize> = HashSet::new();
        let mut i = 1;
        for raw_line in titles_reader.lines() {
            let page = raw_line.unwrap();
            if page_regex.is_match(&page) {
                let low_page = page.to_lowercase();
                if !stop_words.contains(&low_page) {
                    page_map.insert(low_page, i);
                    vertex_set.insert(i);
                }
            }
            i += 1;
        }
        page_map.shrink_to_fit();

        let mut new_vertex_map: HashMap<usize, usize> = HashMap::new();
        for (i, vid) in vertex_set.iter().enumerate() {
            new_vertex_map.insert(*vid, i);
        }

        let mut vertex_page_map: HashMap<usize, String> = HashMap::new();
        for (page, vid) in &page_map {
            let new_vid = new_vertex_map[vid];
            vertex_page_map.insert(new_vid, page.to_string());
        }

        let n_vertexes = new_vertex_map.len();

        println!("Found {} vertexes", n_vertexes);
        let mut adjacency_list: Vec<Vec<usize>> = Vec::with_capacity(n_vertexes);
        let mut rev_adjacency_list: Vec<Vec<usize>> = Vec::with_capacity(n_vertexes);
        for _ in 0..n_vertexes {
            adjacency_list.push(Vec::new());
            rev_adjacency_list.push(Vec::new());
        }

        let links_file = File::open("data/links-simple-sorted.txt").unwrap();
        let links_reader = BufReader::new(&links_file);
        let mut n = 0;
        for raw_line in links_reader.lines() {
            let line = raw_line.unwrap();
            let parsed: Vec<&str> = line.split(':').collect();
            assert!(parsed.len() == 2);
            let source = parsed[0].parse::<usize>().unwrap();
            if vertex_set.contains(&source) {
                let edges = parsed[1].split_whitespace().map(|e| e.parse::<usize>().unwrap());
                let new_u = new_vertex_map[&source];
                let ref mut source_list = adjacency_list[new_u];
                for e in edges {
                    if vertex_set.contains(&e) {
                        let new_v = new_vertex_map[&e];
                        source_list.push(new_v);
                        let ref mut rev_source_list = rev_adjacency_list[new_v];
                        rev_source_list.push(new_u);
                        n += 1
                    }
                    source_list.shrink_to_fit();
                }
            }
        }
        for list in &mut rev_adjacency_list {
            list.shrink_to_fit();
        }
        println!("Found {} links", n);
        WikiGraph {
            adjacency_list: adjacency_list,
            rev_adjacency_list: rev_adjacency_list,
            page_map: page_map,
            vertex_set: vertex_set,
            new_vertex_map: new_vertex_map,
            vertex_page_map: vertex_page_map
        }
    }

}

pub fn path_lengths(wiki_graph: &WikiGraph,
                    sources: &Vec<usize>,
                    destinations: &Vec<usize>,
                    max_distance: usize) -> Vec<(usize, usize, usize)> {
    let mut final_distances: Vec<(usize, usize, usize)> = Vec::new();
    let mut forward_fringe: Vec<usize> = Vec::new();
    let mut backward_fringe: Vec<usize> = Vec::new();
    let mut forward_distances: HashMap<usize, usize> = HashMap::new();
    let mut backward_distances: HashMap<usize, usize> = HashMap::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let ref adjacency_list = wiki_graph.adjacency_list;
    let ref rev_adjacency_list = wiki_graph.rev_adjacency_list;
    for s in sources {
        forward_fringe.clear();
        forward_fringe.push(*s);
        forward_distances.insert(*s, 0);
        let mut forward_dist = 0;
        for d in destinations {
            let mut backward_dist = 0;
            backward_fringe.clear();
            backward_distances.clear();
            backward_fringe.push(*d);
            backward_distances.insert(*d, 0);
            let mut fringe = Vec::new();
            let mut distance: Option<usize> = None;
            while forward_dist + backward_dist < max_distance && distance == None {
                fringe.clear();
                if forward_dist <= backward_dist {
                    // Since the forward distances and fringe are re-used, can't simply
                    // early terminate here to maintain correctness
                    fringe.append(&mut forward_fringe);
                    forward_dist += 1;
                    for node in &fringe {
                        if backward_distances.contains_key(node) {
                            distance = Some(backward_distances[node] + forward_dist);
                            // Cannot early stop to keep forward state valid
                        }
                        for new_node in &adjacency_list[*node] {
                            forward_fringe.push(*new_node);
                        }
                        forward_distances.insert(*node, forward_dist);
                    }
                } else {
                    // Since backward distances and fringes are not re-used, early stopping is fine
                    // here
                    fringe.append(&mut backward_fringe);
                    backward_dist += 1;
                    for node in &fringe {
                        if forward_distances.contains_key(node) {
                            distance = Some(forward_distances[node] + backward_dist);
                            // early stop since backward state doesn't matter
                            break;
                        }
                        for new_node in &rev_adjacency_list[*node] {
                            backward_fringe.push(*new_node);
                        }
                        backward_distances.insert(*node, backward_dist);
                    }
                }
            }
            final_distances.push((*s, *d, distance.unwrap_or(usize::max_value())));
        }
    }

    final_distances
}

#[cfg(test)]
mod tests {
    use super::{load_stopwords, tokenize};
    use itertools::assert_equal;

    #[test]
    fn test_load_stopwords() {
        let stop_words = load_stopwords();
        assert!(stop_words.contains("the"));
        assert!(stop_words.contains("here"));
    }

    #[test]
    fn test_tokenize() {
        let text = "Here is a test, make sure to remove the command And the period. ALso remove stop words";
        let stop_words = load_stopwords();
        let tokens = tokenize(text, &stop_words);
        for t in &tokens {
            assert_eq!(t, &t.to_lowercase());
        }
        assert_equal("test make sure remove command period also remove stop words".split_whitespace(), tokens);
    }
}
