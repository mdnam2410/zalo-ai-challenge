echo "Saving docker image..."
if sudo docker save -o zac2022.tar.gz zac2022; then
    FILESIZE=$(stat --printf="%s" zac2022.tar.gz | numfmt --to=iec)
    echo "Saved, file size $FILESIZE"

    echo "Generating checksum..."
    md5sum zac2022.tar.gz > zac2022.tar.gz.md5

    echo "Verifying checksum..."
    md5sum -c zac2022.tar.gz

else
    echo "Save failed"
fi
