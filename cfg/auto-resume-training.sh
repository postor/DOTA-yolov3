darknet detector train dota.data dota-yolov3-tiny.cfg
while true
do
darknet detector train dota.data dota-yolov3-tiny.cfg backup/dota-yolov3-tiny.backup
sleep 1
done
