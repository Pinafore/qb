for ii in dylan-thomas william-butler-yeats robert-frost thomas-stearns-eliot wallace-stevens william-carlos-williams rainer-maria-rilke seamus-heaney-3 sylvia-plath ezra-pound allen-ginsberg david-herbert-lawrence langston-hughes pablo-neruda william-butler-yeats emily-dickinson wystan-hugh-auden
do
    python get_poem.py $ii
    cp $ii.pkl cache.pkl
done

python copy_to_source.py
