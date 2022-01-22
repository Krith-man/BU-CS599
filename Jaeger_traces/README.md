## Commands written for Assignment IV task 4.

### Commands on local terminal

#### Pass source code to head node
```bash
$ scp -i cloud.key -r ~/Documents/cs599l1-2021 ubuntu@128.31.24.160:/root
```
#### Enter head-node instance from local terminal with local ssh porting
```bash
ssh -i cloud.key ubuntu@128.31.24.160 -Y -L 16686:localhost:16686 
```
<br/><br/>
### Commands on head-node
#### Create virtual environment with python 3.9
```bash
sudo apt update
sudo apt install python3.9
sudo  apt-get install python3.9-dev python3.9-venv
mkdir myenv
cd myenv
python3.9 -m venv venv
source venv/bin/activate
```
#### install program requirements written in requirement.txt file
```bash
pip install -r requirements.txt
```
#### install ray dependencies 
```bash
pip install -U "ray[default]"
```
#### install jaeger dependencies (including docker)
```bash
pip install jaeger-client
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
sudo apt install docker-ce
sudo systemctl status docker 
```
#### Change permissions and start your docker container for JaegerUI on head node.
```bash
sudo chmod 666 /var/run/docker.sock
sudo docker run -d --name jaeger -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p 5775:5775/udp -p 6831:6831/udp -p 6832:6832/udp -p 5778:5778 -p 16686:16686 -p 14268:14268 -p 14250:14250 -p 9411:9411 jaegertracing/all-in-one:latest
```
#### Start ray application 
```bash
ray start --head --port=6379
```
#### Move to worker nodes 
```bash
ssh -i cloud.key ubuntu@10.0.0.31
ssh -i cloud.key ubuntu@10.0.0.137
ssh -i cloud.key ubuntu@10.0.0.78
ssh -i cloud.key ubuntu@10.0.0.235
```
##### start the python script 
```bash
python assignment_4.py --assignment 4 --task 2 --uid 0 --mid 3
```


<br/><br/>
### Commands on each worker node.
#### Create virtual environment with python 3.9
```bash
sudo apt update
sudo apt install python3.9
sudo  apt-get install python3.9-dev python3.9-venv
mkdir myenv
cd myenv
python3.9 -m venv venv
source venv/bin/activate
```
#### install ray dependencies 
```bash
pip install -U "ray[default]"
```
#### install jaeger dependencies (including docker)
```bash
pip install jaeger-client
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
sudo apt install docker-ce
sudo systemctl status docker 
```
#### Change permissions and start your docker container for JaegerUI on worker node.
```bash
sudo chmod 666 /var/run/docker.sock
sudo docker run --rm -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp -p5778:5778/tcp jaegertracing/jaeger-agent --reporter.grpc.host-port=10.0.0.73:14250
```
#### Ray start node after ray's initialization to head node. 
```bash
ray start --address='10.0.0.73:6379' --redis-password='5241590000000000'
```
