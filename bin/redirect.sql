SELECT
p.page_title AS source_page,
r.rd_title AS dest_page
FROM page p
INNER JOIN (SELECT rd_title, rd_from FROM redirect WHERE rd_namespace=0) r ON p.page_id=r.rd_from
WHERE page_namespace=0
INTO OUTFILE '/var/lib/mysql/redirect.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
