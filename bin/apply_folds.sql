UPDATE questions SET fold = "train";
UPDATE questions SET fold = "dev" where naqt > 0;
UPDATE questions SET fold = "devtest" where tournament LIKE "2011-12 High School Championship%";
UPDATE questions SET fold = "test" where tournament LIKE "2012-13 High School Championship%";
