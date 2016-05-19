mkdir -p $1

for ii in `seq 1 $2`
do
   curl http://www.poemhunter.com/$1/poems/page-$ii | grep -oh "poem/[0-9a-zA-Z\-]*" > $1/$ii
   sleep 5
done