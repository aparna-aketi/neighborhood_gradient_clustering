### IID (Plot1)
python trainer.py --optimizer ngc --arch=cganet --world_size=5 --lr=0.01 --gamma=1 --alpha 1.0 --skew=0 --graph ring --normtype evonorm --neighbors 2 --batch-size=160 --epoch 300 --momentum=0.9 --steplr --data-dir ./data --device 4
cd ./outputs
python dict_to_csv.py --optimizer ngc --arch=cganet --world_size=5 --lr=0.01 --gamma=1 --alpha 1.0 --skew=0 --graph ring --norm evonorm
cd ..



python trainer.py --optimizer ngc --arch=cganet --world_size=5 --lr=0.01 --gamma=1 --alpha 1.0 --skew=0 --graph full --normtype evonorm --neighbors 4 --batch-size=160 --epoch 300 --momentum=0.9 --steplr --data-dir ./data --device 4
cd ./outputs
python dict_to_csv.py --optimizer ngc --arch=cganet --world_size=5 --lr=0.01 --gamma=1 --alpha 1.0 --skew=0 --graph full --norm evonorm
cd ..


### Non-IID (Plot2)
python trainer.py --optimizer ngc --arch=cganet --world_size=5 --lr=0.01 --gamma=0.1 --alpha 1.0 --skew=1.0 --graph ring --normtype evonorm --neighbors 2 --batch-size=160 --epoch 300 --momentum=0.9 --steplr --data-dir ./data --device 4
cd ./outputs
python dict_to_csv.py --optimizer ngc --arch=cganet --world_size=5 --lr=0.01 --gamma=0.1 --alpha 1.0 --skew=1.0 --graph ring --norm evonorm
cd ..



python trainer.py --optimizer ngc --arch=cganet --world_size=5 --lr=0.01 --gamma=1 --alpha 1.0 --skew=1.0 --graph full --normtype evonorm --neighbors 4 --batch-size=160 --epoch 300 --momentum=0.9 --steplr --data-dir ./data --device 4
cd ./outputs
python dict_to_csv.py --optimizer ngc --arch=cganet --world_size=5 --lr=0.01 --gamma=1 --alpha 1.0 --skew=1.0 --graph full --norm evonorm
cd ..



### Non-IID (Plot3) different algo
python trainer.py --optimizer cga --arch=cganet --world_size=5 --lr=0.01 --gamma=0.1 --alpha 1.0 --skew=1.0 --graph ring --normtype evonorm --neighbors 2 --batch-size=160 --epoch 300 --momentum=0.9 --steplr --data-dir ./data --device 4
cd ./outputs
python dict_to_csv.py --optimizer cga --arch=cganet --world_size=5 --lr=0.01 --gamma=0.1 --alpha 1.0 --skew=1.0 --graph ring --norm evonorm
cd ..




cd ./outputs
# Figure 1a
python plots.py --nodes 5 --plot_data val_loss --case iid
# Figure 1b
python plots.py --nodes 5 --plot_data val_loss --case noniid
# Figure 1c
python plots.py --nodes 5 --plot_data val_loss --case algo
# Figure 3a
python plots.py --nodes 5 --plot_data epsilon --case algo
# Figure 3b
python plots.py --nodes 5 --plot_data omega --case algo
cd ..