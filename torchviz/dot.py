from collections import namedtuple
from distutils.version import LooseVersion
from graphviz import Digraph
import torch
from torch.autograd import Variable
import warnings



# 名前付きフィールドを持つタプルのサブクラスを生成
Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))
# - name: ノードの名前。これは通常、ノードが表すテンソルまたは操作の名前。
# - inputs: ノードの入力。これは、ノードが依存する他のノードのリスト。
# - attr: ノードの属性。これは、ノードの形状、タイプ、その他の情報を含む辞書。
# - op: ノードが表す操作。これは、ノードが表すPyTorchの操作（例えば、加算、乗算、畳み込みなど）の名前。

# Saved attrs for grad_fn (incl. saved variables) begin with `._saved_*`
SAVED_PREFIX = "_saved_"



# 逆伝播関数（grad_fn）の名前を取得
def get_fn_name(fn, show_attrs, max_attr_chars):
    name = str(type(fn).__name__)
    if not show_attrs:
        return name
    attrs = dict()
    for attr in dir(fn):
        if not attr.startswith(SAVED_PREFIX):
            continue
        val = getattr(fn, attr)
        attr = attr[len(SAVED_PREFIX):]
        if torch.is_tensor(val):
            attrs[attr] = "[saved tensor]"
        elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
            attrs[attr] = "[saved tensors]"
        else:
            attrs[attr] = str(val)
    if not attrs:
        return name
    max_attr_chars = max(max_attr_chars, 3)
    col1width = max(len(k) for k in attrs.keys())
    col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
    sep = "-" * max(col1width + col2width + 2, len(name))
    attrstr = '%-' + str(col1width) + 's: %' + str(col2width)+ 's'
    truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
    params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
    return name + '\n' + sep + '\n' + params



# 指定されたテンソルの計算グラフを作成
def make_dot(var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50):
    
    """ PyTorch autograd graphをGraphvizの表現で生成します。

    ノードが逆方向関数を表す場合、そのノードは灰色になります。
    それ以外の場合、ノードはテンソルを表し、青、オレンジ、または緑のいずれかになります。:
     - 青: grad を必要とする到達可能なリーフ テンソル
           (`.grad` フィールドが `.backward()` 中に設定されるテンソル)
     - オレンジ: カスタム autograd 関数の保存されたテンソルと、組み込みのバックワード ノードによって保存されたテンソル
     - 緑: 出力として渡されるテンソル。
     - 濃い緑色: 出力がビューの場合、その基本テンソルを濃い緑色のノードで表します。

    引数:
        var: 出力テンソル
        params: grad を必要とするノードに名前を追加するための (name, tensor) の辞書
        show_attrs: 逆方向ノードの非テンソル属性を表示するかどうか (PyTorch バージョン >= 1.9 が必要)
        show_saved: カスタム autograd 関数によるものではない保存された tensor ノードを表示するかどうか。
                    カスタム関数用に保存されたテンソル ノードが存在する場合、常に表示されます。
                    (PyTorch バージョン 1.9 以上が必要)
        max_attr_chars: show_attrs が「True」の場合、指定された属性に対して表示する最大文字数を設定します。
    """
    
    if LooseVersion(torch.__version__) < LooseVersion("1.9") and \
        (show_attrs or show_saved):
        warnings.warn(
            "make_dot: showing grad_fn attributes and saved variables"
            " requires PyTorch version >= 1.9. (This does NOT apply to"
            " saved tensors saved by custom autograd functions.)")

    if params is not None:
        
        # paramsのすべての値がVariableインスタンスであることを確認
        assert all(isinstance(p, Variable) for p in params.values())
        
        # params辞書のキーと値を反転した新しい辞書param_mapを作成
        param_map = {id(v): k for k, v in params.items()}
        
    else:
        param_map = {}


    
    # すべてのノードのデフォルトの属性を設定
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.2',
                     fontname='monospace')

    # 空の有向グラフ（Digraph）オブジェクトを作成
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    # スクリプト内で既に処理されたノードを追跡するために使用。set型（重複する要素を持たない、順序を持たない要素）
    seen = set()

    # PyTorchテンソルのサイズを文字列に変換
    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    # テンソルの名前とサイズを含む文字列を生成
    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return '%s\n %s' % (name, size_to_str(var.size()))

    # 指定された逆伝播関数とその入力をグラフに追加。add_base_tensor関数から呼び出される
    def add_nodes(fn):

        # seenに登録があるノードかどうか確認。重複でなければ追加する
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        
        if show_saved:
            for attr in dir(fn):
                if not attr.startswith(SAVED_PREFIX):
                    continue
                val = getattr(fn, attr)
                seen.add(val)
                attr = attr[len(SAVED_PREFIX):]
                if torch.is_tensor(val):
                    dot.edge(str(id(fn)), str(id(val)), dir="none")
                    dot.node(str(id(val)), get_var_name(val, attr), fillcolor='orange')
                if isinstance(val, tuple):
                    for i, t in enumerate(val):
                        if torch.is_tensor(t):
                            name = attr + '[%s]' % str(i)
                            dot.edge(str(id(fn)), str(id(t)), dir="none")
                            dot.node(str(id(t)), get_var_name(t, name), fillcolor='orange')

        # fnがvariable属性を持っている場合（つまり、fnがテンソルの逆伝播関数である場合）に特定のコードブロックを実行
        # テンソルとその逆伝播関数をグラフに追加するための手段
        if hasattr(fn, 'variable'):
            # grad_accumulator の場合、`.variable` のノードを追加します
            var = fn.variable
            seen.add(var)
            dot.node(str(id(var)), get_var_name(var), fillcolor='lightblue')
            dot.edge(str(id(var)), str(id(fn)))

        # この grad_fn のノードを追加します
        dot.node(str(id(fn)), get_fn_name(fn, show_attrs, max_attr_chars))

        # fnがnext_functions属性を持っている場合（つまり、fnが他の逆伝播関数に依存している場合）に特定のコードブロックを実行
        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(fn)))
                    add_nodes(u[0])

        """ 
        注: これは、pytorch0.2 で .saved_tensors を表示していました。
        しかし、ATen に移行され、Variable-Tensor がマージされたため機能しなくなりました。
        また、これはカスタム autograd 関数ではまだ機能することに注意してください。
        """
        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                seen.add(t)
                dot.edge(str(id(t)), str(id(fn)), dir="none")
                dot.node(str(id(t)), get_var_name(t), fillcolor='orange')

    # 指定されたテンソルとその基本テンソルをグラフに追加
    def add_base_tensor(var, color='darkolivegreen1'):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if (var.grad_fn):
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color='darkolivegreen3')
            dot.edge(str(id(var._base)), str(id(var)), style="dotted")


    # 複数の出力を処理する
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)

    return dot



# 未使用
def make_dot_from_trace(trace):
    """
    この機能は、次の pytorch コアでは利用できません。
    https://pytorch.org/docs/stable/tensorboard.html
    """
    # from tensorboardX
    raise NotImplementedError("This function has been moved to pytorch core and "
                              "can be found here: https://pytorch.org/docs/stable/tensorboard.html")



# グラフに含まれるノードとエッジの数に基づいてグラフのサイズを変更
def resize_graph(dot, size_per_element=0.15, min_size=12):
    """
    グラフに含まれるコンテンツの量に応じてグラフのサイズを変更します。
    グラフをその場で変更します。
    """
    # ノードとエッジのおおよその数を取得する
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
