# Bachelorarbeit 

Codegrundlage für die Bachelorarbeit *Einfluss Approximierter Gradienten auf gradientenbasierten Feature-Approximationsmethoden* von Benjamin Weis.



## Überblick

In diesem Repository befinden sich alle Notebooks, welche für die Generierung der approximierten Attributionen und ihrer Auswertung genutzt wurden. 


## Installation

### Voraussetzungen
Alle benötigten Packages sind in der requirements.txt aufgelistet.

### Schritt-für-Schritt Anleitung

```bash
# Repository klonen
git clone https://github.com/Benomin/Approx_Grad_Methoden.git

pip install -r requirements.txt
```

## Verwendung
Um eine Approximierte Attribution zu generieren, wird wie folgt vorgegangen:
```python
from approx_attributes import *


model_CNN = CNN().to(device)

state_dict = torch.load("Models/CifarModelCnn.pth", map_location=torch.device(device))

model_CNN.load_state_dict(state_dict)
model_CNN.eval()

phi = approx_attributes(model_CNN)

img, label = dataset[index]

##################################################
######### Für eine Salienzkarte dann: ############
##################################################

attribution_map = phi.grad_approx(h=0.1, [img.unsqueeze(0)], target=label)

##################################################
######### Für Gradient x Input dann: #############
##################################################

attribution_map = phi.aa.grad_x_i_approx(h=0.1, [img.unsqueeze(0)], target=label)


##################################################
######## Für Integrated Gradients dann: ##########
##################################################

attribution_map = phi.grad_approx(h=0.1, [img.unsqueeze(0)], target=label).reshape(3, 32, 32)


"""
h: Schrittweite
target: Ausgabe, für die eine Attribution berechnet werden soll
"""
```
Hierbei können natürlich auch Attributionen für mehrere Eingaben generiert werden.

Das Jupyter Notebook Grafics.ipynb enthält den für Abschnitt 3.1 bis 3.3 Relevannten Code.
In diesem werden die 3 verwendeten Modelle und der Datensatz passend geladen und jeweils für die Daten aus *Approximated Data* Boxplots erstellt. Die daten, aus denen die Boxplots erstellt wurden , wurden mithilfe der funktion gen_data() erstellt, eine beispielhafte Anwendung befindet sich im Code.


Die Datei Synthetic Data.ipynb enthält den Code für den synthetischen Datensatz und die auswertung dieser Daten. Die verwendetetn generierten Datensätze und die dazugehörigen Models sind in den Ordnern *Synthetic_Data* und *Models* zu finden.

