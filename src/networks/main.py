from .mlp import MLP, MLP_Autoencoder
from .convAutoencoder import ConvEncoder, Conv_Autoencoder
from .convAutoencoder_MLP import ConvEncoder_MLP, Conv_Autoencoder_MLP

def build_network(net_name, ae_net=None):
    """Builds the neural network."""

    implemented_networks = ('HDFS_mlp', 'HDFS_MIMO_conv_mlp',
                            'BGL_mlp', 'BGL_MIMO_conv_mlp',
                            'LDAP_mlp', 'LDAP_MIMO_conv_mlp')
    assert net_name in implemented_networks

    net = None

    if net_name in ('HDFS_mlp', 'BGL_mlp', 'LDAP_mlp'):
        net = MLP(x_dim=64, h_dims=[32, 16], rep_dim=64, bias=False)
        
    if net_name == 'HDFS_MIMO_conv_mlp':
        # HDFS 64*64
        net = ConvEncoder_MLP(encoded_space_dim=64, conv_dim=32*4*4,unflattened_size=(32,4,4), rep_dim=64) 
        
    if net_name == 'BGL_MIMO_conv_mlp':
        # BGL 128*128
        net = ConvEncoder_MLP(encoded_space_dim=64, conv_dim=32*8*8,unflattened_size=(32,8,8), rep_dim=64) 
        
    if net_name == 'LDAP_MIMO_conv_mlp':
        # LDAP 32*32
        net = ConvEncoder_MLP(encoded_space_dim=64, conv_dim=32*2*2,unflattened_size=(32,2,2), rep_dim=64) 

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('HDFS_mlp', 'HDFS_MIMO_conv_mlp',
                            'BGL_mlp', 'BGL_MIMO_conv_mlp',
                            'LDAP_mlp', 'LDAP_MIMO_conv_mlp')

    assert net_name in implemented_networks

    ae_net = None

    if net_name in ('HDFS_mlp', 'BGL_mlp', 'LDAP_mlp'):
        ae_net = MLP_Autoencoder(x_dim=64, h_dims=[32, 16], rep_dim=64, bias=False)
        
    if net_name == 'HDFS_MIMO_conv_mlp':
        # HDFS 64*64
        ae_net = Conv_Autoencoder_MLP(encoded_space_dim=64,conv_dim=32*4*4,unflattened_size=(32,4,4), rep_dim=64) # 64 
        
    if net_name == 'BGL_MIMO_conv_mlp':
        # BGL 128*128
        ae_net = Conv_Autoencoder_MLP(encoded_space_dim=64,conv_dim=32*8*8,unflattened_size=(32,8,8), rep_dim=64) #  128UP
        
    if net_name == 'LDAP_MIMO_conv_mlp':
        # LDAP 32*32
        ae_net =  Conv_Autoencoder_MLP(encoded_space_dim=64,conv_dim=32*2*2,unflattened_size=(32,2,2), rep_dim=64) # 32 LDAP 

    print(ae_net)
    return ae_net
