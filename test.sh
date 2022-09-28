python trainer.py  --data-dir ../data   --lr=0.01  --batch-size=160  --world_size=5 --skew=1 --gamma=0.1 --normtype evonorm --optimizer ngc --epoch 100 --arch=cganet --momentum=0.9 --alpha 1.0 
cd ./outputs
python dict_to_csv.py --norm evonorm --lr 0.01 --gamma 0.1 --arch cganet --world_size 5 --optimizer ngc --alpha 1.0 --skew 1 --graph ring
cd ..

python trainer.py  --data-dir ../data   --lr=0.01  --batch-size=160  --world_size=5 --skew=1 --gamma=0.1 --normtype evonorm --optimizer cga --epoch 100 --arch=cganet --momentum=0.9 --alpha 1.0
cd ./outputs
python dict_to_csv.py --norm evonorm --lr 0.01 --gamma 0.1 --arch cganet --world_size 5 --optimizer cga --alpha 1.0 --skew 1 --graph ring
cd ..

python trainer.py  --data-dir ../data   --lr=0.01  --batch-size=160  --world_size=5 --skew=1 --gamma=0.1 --normtype evonorm --optimizer compngc --epoch 100 --arch=cganet --momentum=0.9 --alpha 1.0 
cd ./outputs
python dict_to_csv.py --norm evonorm --lr 0.01 --gamma 0.1 --arch cganet --world_size 5 --optimizer compngc --alpha 1.0 --skew 1 --graph ring
cd ..

python trainer.py  --data-dir ../data   --lr=0.01  --batch-size=160  --world_size=5 --skew=1 --gamma=0.1 --normtype evonorm --optimizer compcga --epoch 100 --arch=cganet --momentum=0.9 --alpha 1.0
cd ./outputs
python dict_to_csv.py --norm evonorm --lr 0.01 --gamma 0.1 --arch cganet --world_size 5 --optimizer compcga --alpha 1.0 --skew 1 --graph ring
cd ..

python trainer.py  --data-dir ../data   --lr=0.1  --batch-size=160  --world_size=5 --skew=1 --gamma=1 --normtype evonorm --optimizer d-psgd --epoch 100 --arch=cganet --momentum=0.0 --alpha 1.0
cd ./outputs
python dict_to_csv.py --norm evonorm --lr 0.1 --gamma 1 --arch cganet --world_size 5 --optimizer d-psgd --alpha 1.0 --skew 1 --graph ring
cd ..

cd ./outputs
python plots.py --case test --plot_data val_loss --node 5
cd ..


