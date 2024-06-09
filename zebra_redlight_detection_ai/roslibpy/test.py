import roslibpy
# docker config
# docker run -p 9090:9090 -v /:/dockerfile -it --env="DISPLAY=$DISPLAY"  --name=roslibpy ros:melodic  /bin/bash
# docker run --network host -v /:/dockerfile -it --env="DISPLAY=$DISPLAY"  --name=roslibpy ros:melodic  /bin/bash
# docker start roslibpy
# docker attach roslibpy
# docker exec -it roslibpy bash -c "source ~/.bashrc; bash"
# docker exec -it roslibpy bash -c "source /etc/bash.bashrc; source /root/.bashrc; bash"

client = roslibpy.Ros(host='localhost', port=9090)
client.run()
print('Is ROS connected?', client.is_connected)

talker = roslibpy.Topic(client, '/chatter', 'std_msgs/String')
talker.publish(roslibpy.Message({'data': 'Hello, ROS!'}))

print('Message sent to ROS!')
client.terminate()