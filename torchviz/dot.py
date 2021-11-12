from collections import namedtuple
from distutils.version import LooseVersion
from graphviz import Digraph
import torch
from torch.autograd import Variable
import warnings


Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"



class Dot_node(object):
    def __init__(self, var, id_: str='', name_: str='', index=None, **attrs):
        '''
        Attributes:
            var: a variable (Tensor with requires_grad=True).
            name: id of the object
            label: the name in the computational graph, that is type(node).__name__
            index: the position in the dot graph
        Methods:
            get_fn_name: return the the variable/tensor name. I
                f show_attrs=True it also returns other attributes of the variable/tensor.
        '''
        self.var = var       
        self.id =  str(id(self))
        self.name = type(self.var).__name__
        self.index = index
        self.attrs = dict()
        
        for attr in dir(self.var):
            if not attr.startswith(SAVED_PREFIX):
                continue
            val = getattr(self.var, attr)
            attr = attr[len(SAVED_PREFIX):]
            if torch.is_tensor(val):
                self.attrs[attr] = "[saved tensor]"
            elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
                self.attrs[attr] = "[saved tensors]"
            else:
                self.attrs[attr] = str(val)
        
    def get_fn_name(self, show_attrs, max_attr_chars, removeBackward=False):
        name = self.name
        if removeBackward:
            name = name.replace('Backward','')
                
        if not self.attrs:
            return name
        if self.attrs.get('index') is not None:
           name += '_'+self.attrs.get('index')
         
        if not show_attrs:
            return name
          
        max_attr_chars = max(max_attr_chars, 3)
        col1width = max(len(k) for k in self.attrs.keys())
        col2width = min(max(len(str(v)) for v in self.attrs.values()), max_attr_chars)
        sep = "-" * max(col1width + col2width + 2, len(name))
        attrstr = '%-' + str(col1width) + 's: %' + str(col2width)+ 's'
        truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
        params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in self.attrs.items())
        return name + '\n' + sep + '\n' + params
    
    

class Dot_graph(object):
    def __init__(self, var, model=None, instance=None, params=None, show_attrs=False, show_saved=False, max_attr_chars=50):
        """ 
        Produces Graphviz representation of PyTorch autograd graph.
        If a node represents a backward function, it is gray. Otherwise, the node
        represents a tensor and is either blue, orange, or green:
         - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
             fields will be populated during `.backward()`)
         - Orange: saved tensors of custom autograd functions as well as those
             saved by built-in backward nodes
         - Green: tensor passed in as outputs
         - Dark green: if any output is a view, we represent its base tensor with
             a dark green node.
        
        Attributes:
            var: output tensor
            model: a class that heritages from nn.module
            instance: a variable or a tensor
            params: dict of (name, tensor) to add names to node that requires grad
            show_attrs: whether to display non-tensor attributes of backward nodes
                (Requires PyTorch version >= 1.9)
            show_saved: whether to display saved tensor nodes that are not by custom
                autograd functions. Saved tensor nodes for custom functions, if
                present, are always displayed. (Requires PyTorch version >= 1.9)
            max_attr_chars: if show_attrs is `True`, sets max number of characters
                to display for any given attribute.
            nodes_select: nodes of type 'select' (that have the attribute 'index').
            seen: set of nodes already visited
            param_map: dictionary with the dot.nodes that encapsulate the parameters of the model (weights, biases, etc.0)
            output_map: dictionary with the output dot.nodes
            input_map: dictionary with the input dot.nodes
            
            
         Methods:
            make_dot: show the backward computational graph (attributes and saved variables)
            add_base_tensor: add tensors passed in as outputs
            add_nodes: add a node that represents a backward function
            get_var_name: return the name of a variable/tensor.
            size_to_str: transform the size of a variable/tensor to a custom string format.
            resize_graph: resize the graph according to how much content it contains. 
                Modify the computational graph in place.
        """
        
        self.var = var
        
        self.model = model
        self.weights = None
        self.biases = None
        self.instance = instance
        self.params = params
        self.show_attrs=show_attrs
        self.show_saved = show_saved
        self.max_attr_chars = max_attr_chars
        
        
        self.nodes_select = {}
        #self.edges = {}
        
        self.node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='10',
                         ranksep='0.1',
                         height='0.2',
                         fontname='monospace')
        self.dot = None
        self.seen = set()
        self.param_map = {}
        self.output_map = {}
        self.input_map = {}
       
        
        # adding the parameters of the model (weights and biases)
        if self.params is not None:
            assert all(isinstance(p, Variable) for p in self.params.values())
            self.param_map = {id(v): k for k, v in self.params.items()}
        else:
            self.param_map = {}
            if self.model is not None:
                params_iter = self.model.parameters()
                try:
                    self.weights = next(params_iter)
                    self.param_map[id(self.weights)] = 'weights'
                except:
                    pass
                
                try:
                    self.biases = next(params_iter)
                    self.param_map[id(self.biases)] = 'biases'
                except:
                    pass
            if self.instance is not None :
                try:
                    self.input_map[id(self.instance)] = 'input'
                except:
                    pass
                
        # handle multiple outputs
        if isinstance(self.var, tuple):
            for v in self.var:
                self.output_map[id(v)] = 'output'
        else:
            self.output_map[id(self.var)] = 'output'
            

    
    def make_dot(self, simplified=False):
        
        self.dot = Digraph(node_attr=self.node_attr, graph_attr=dict(size="12,12"))
        
        if LooseVersion(torch.__version__) < LooseVersion("1.9") and \
            (self.show_attrs or self.show_saved):
            warnings.warn(
                "make_dot: showing grad_fn attributes and saved variables"
                " requires PyTorch version >= 1.9. (This does NOT apply to"
                " saved tensors saved by custom autograd functions.)")

    
        # handle multiple outputs
        if isinstance(self.var, tuple):
            for v in self.var:
                self.add_base_tensor(v, simplified=simplified)
        else:
            self.add_base_tensor(self.var, simplified=simplified)
    
        self.resize_graph()
        
        # reset some parameters
        self.seen = set()    
        return self.dot
    
    
    def add_base_tensor(self, var, color='darkolivegreen1', simplified=False):
        if var in self.seen:
            return
        self.seen.add(var)
        self.dot.node(str(id(var)), self.get_var_name(var), fillcolor=color)
        try:
            isGrad_fn = var.grad_fn
        except AttributeError:
            isGrad_fn = False
            print('You are trying to plot the module without a valid input. It doesn\'t have the attribute grad_fn')
        
        # add the node connected to the variable except if the simplified model is requested and the next node is of of type 'View'
        if isGrad_fn:
            u = var.grad_fn
            if not simplified:
                self.add_nodes(u,simplified)
                self.dot.edge(str(id(u)), str(id(var)))
            else:
                if 'View' not in type(u).__name__:
                    self.add_nodes(u,simplified)
                    self.dot.edge(str(id(u)), str(id(var)))
                elif 'View' in type(u).__name__:
                    v = u.next_functions[0][0]  
                    self.add_nodes(v,simplified)
                    self.dot.edge(str(id(v)), str(id(var)))
                    self.seen.add(u)

    
    def add_nodes(self, fn, simplified=False):
        assert not torch.is_tensor(fn)
        if fn in self.seen:
            return
        self.seen.add(fn)
        node_ = Dot_node(fn)
        
        # in the simplified version the 'Select' prefix is replaced by the parameter name (weights, biases, etc.)
        if simplified:
            if node_.attrs.get('index') is not None:
                #self.nodes_select[str(id(fn))] = self.param_map[id(fn.next_functions[0][0].variable)]
                u = fn.next_functions[0]
                if u[0] is not None:
                    if hasattr(u[0], 'variable'):
                        var = u[0].variable
                        if id(var) in self.param_map:
                            type_param = self.param_map[id(var)][0]
                            self.nodes_select[str(id(fn))] = type_param
                        elif id(var) in self.input_map:
                            type_param = self.input_map[id(var)][0]
                            self.nodes_select[str(id(fn))] = type_param
            
        if self.show_saved:
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(fn, attr)
                self.seen.add(val)
                attr = attr[len(SAVED_PREFIX):]
                if torch.is_tensor(val):
                    self.dot.edge(str(id(fn)), str(id(val)), dir="none")
                    self.dot.node(str(id(val)), self.get_var_name(val, attr), fillcolor='orange')
                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
                            name = attr + '[%s]' % str(i)
                            self.dot.edge(str(id(fn)), str(id(t)), dir="none")
                            self.dot.node(str(id(t)), self.get_var_name(t, name), fillcolor='orange')
    
        
        # add the node for this grad_fn (the input nodes)
        if hasattr(fn, 'variable'):
            # if grad_accumulator, add the node for `.variable`
            var = fn.variable
            self.seen.add(var)
            self.dot.node(str(id(fn)), self.get_var_name(var), fillcolor='lightblue')
        
        else:
            if not simplified:
                self.dot.node(str(id(fn)), node_.get_fn_name(self.show_attrs, self.max_attr_chars))  
            else:
                node_name = node_.get_fn_name(self.show_attrs, self.max_attr_chars,removeBackward=True)
                if str(id(fn)) in self.nodes_select:
                    node_name = node_name.replace('Select',self.nodes_select.get(str(id(fn))))
                else:
                    node_name = node_name.replace('0','')
                self.dot.node(str(id(fn)), node_name)
        
    
        # recurse
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    if not simplified:
                        self.dot.edge(str(id(u[0])), str(id(fn)))
                        self.add_nodes(u[0])
                    else:
                        # in the simplified version the view nodes are skipped
                        if 'View' not in type(u[0]).__name__:
                            self.dot.edge(str(id(u[0])), str(id(fn)))
                            self.add_nodes(u[0], simplified)
                        elif hasattr(u[0], 'next_functions') :
                            self.seen.add(u[0])
                            v = u[0].next_functions[0][0]  
                            self.dot.edge(str(id(v)), str(id(fn)))
                            self.add_nodes(v, simplified)
    
        # note: this used to show .saved_tensors in pytorch0.2, but stopped
        # working* as it was moved to ATen and Variable-Tensor merged
        # also note that this still works for custom autograd functions
        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                self.seen.add(t)
                self.dot.edge(str(id(t)), str(id(fn)), dir="none")
                self.dot.node(str(id(t)), self.get_var_name(t), fillcolor='orange')
                    
    
    def get_var_name(self, var, name=None):
        if not name:
            if id(var) in self.param_map:
                name = self.param_map[id(var)]
            else:
                if id(var) in self.output_map:
                    name = self.output_map[id(var)]
                elif id(var) in self.input_map:
                    name = self.input_map[id(var)]
                else:
                    name = ''
        try:
            size = self.size_to_str(var.size())
        except AttributeError:
            size = ''
            print('You are trying to plot the module without a valid input. It doesn\'t have the attribute size')
        return '%s\n %s' % (name, size)
        
    def size_to_str(self, size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'
    
    def make_dot_from_trace(trace):
        """ This functionality is now available in pytorch core at
        https://pytorch.org/docs/stable/tensorboard.html
        """
        # from tensorboardX
        raise NotImplementedError("This function has been moved to pytorch core and "
                              "can be found here: https://pytorch.org/docs/stable/tensorboard.html")
    
    def resize_graph(self, size_per_element=0.15, min_size=12):
        # Get the approximate number of nodes and edges
        num_rows = len(self.dot.body)
        content_size = num_rows * size_per_element
        size = max(min_size, content_size)
        size_str = str(size) + "," + str(size)
        self.dot.graph_attr.update(size=size_str)
