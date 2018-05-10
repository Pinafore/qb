SELECT
cl_from AS page_id,
cl_to AS category
FROM categorylinks
INTO OUTFILE '/var/lib/mysql/categorylinks.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
