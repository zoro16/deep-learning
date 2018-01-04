#!/bin/bash
set -m

counter=3
while [ $counter -le 0120 ]
do
    # num=`%04d $counter`
    # echo $num  
    FILE=$(printf %04d $counter)
   curl 'https://hub.coursera-notebooks.org/user/qaaqdkjveiaeodzmoenidk/files/week3/Car%20detection%20for%20Autonomous%20Driving/images/'$FILE'.jpg' -H 'Host: hub.coursera-notebooks.org' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' -H 'Accept-Language: en,ms;q=0.5' --compressed -H 'Referer: https://hub.coursera-notebooks.org/user/qaaqdkjveiaeodzmoenidk/view/week3/Car%20detection%20for%20Autonomous%20Driving/images/0002.jpg' -H 'Cookie: jupyter-hub-token-qaaqdkjveiaeodzmoenidk="2|1:0|10:1515032637|40:jupyter-hub-token-qaaqdkjveiaeodzmoenidk|44:NTcwZmUzMDM5ODQ4NGQ1ZmI2OWMxNTg5YmQwZmY5ZWI=|12a7ea52461dd87fc6851c46955fd65b9b50b95c3227349b3fc91ffd2662c498"; AWSALB=DpV5F4XwK4KIaa1spk4aoInO/kun/bG1gJXHFM89mhkbmBCcVxoZd6D+cIZ5s4GwP5rkky/1BqststOO81gVupYuN/pzrdCU9WJsBfU2SZuG0Z2A4lVk/jP5Jy3P; _xsrf=2|cab9932d|bb6440bf3c6702b24429961c47fb4678|1513776937' -H 'Connection: keep-alive' -H 'Upgrade-Insecure-Requests: 1' -H 'Cache-Control: max-age=0' --output ~/$FILE.jpg
    ((counter++))
done
echo All done
