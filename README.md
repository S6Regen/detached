#Detached switch state neural networks.

In a ReLU neural network and similar the switch state predicate is bound to the activation funtion.

Ie. predicate is x>=0.  true=on , false=off.


What if you replace the internal predicates with an external ones, for example using a locality sensitive hash
of the input vector to supply the true false terms?
Would that even work?
What are the further consequences?

It does seem to work fine just on a first try.  I have to check what effect it has on generalization etc. 
