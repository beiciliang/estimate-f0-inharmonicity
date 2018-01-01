# estimate-f0-inharmonicity
Estimate the fundamental frequency `F0` and inharmonicity coefficient `B` of an isolated piano note

The code implements the estimation approach in

```
F. Rigaud, B. David, and L. Daudet, 
“A parametric model and estimation techniques for the inharmonicity and tuning of the piano” ,
The Journal of the Acoustical Society of America, vol. 133, no. 5, pp. 3017–3118, 2013.
```

The paper describes a robust new algorithm based on the Non-negative Matrix Factorization (NMF) frameworks in order to finely estimate `(F0,B)` from isolated notes.

### How to run
1. Clone the directory
```
 $ git clone https://github.com/beiciliang/estimate-f0-inharmonicity.git
```
2. Install the requirements using pip
```
$ cd estimate-f0-inharmonicity
$ pip install --user -r requirements.txt
```
3. Run the python file
```
$ python estimate-f0-inharmonicity.py [your path containing .wav file of an isolated piano note] [The midi number of the note in the given wav file]
```
using the data provided in this repo as an example:
```
$ python estimate-f0-inharmonicity.py ./data/IS-v96-m60.wav 60 
```
it will return:
```
For note MIDI-No.60:
the estimated fundamental frequency is 262.399515937
the estimated inharmonicity coefficient is 0.000326411619295
```

The source code could also calculate the first 30th partials and their amplitude based on the estimated `(F0,B)`. Feel free to use it if it's useful for your piano-related projects.

P.S. A matlab implementation is provided at [Tian Cheng's soundsoftware repository](https://code.soundsoftware.ac.uk/projects/inharmonicityestimation).
