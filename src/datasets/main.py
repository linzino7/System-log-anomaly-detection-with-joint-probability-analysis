from .graphseqlogs import graphseqLogsSADDataset
from .networkseqlogs import networkseqLogsSADDataset


def load_dataset(dataset_name, data_path, net_name, normal_class, known_outlier_class, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('HDFS', 'LDAP', 'BGL')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name in ('network_seq'):
    #TODO　
        dataset = networkseqLogsSADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name in ('HDFS', 'LDAP', 'BGL'):
    #TODO　
        dataset = graphseqLogsSADDataset(root=data_path,
                                dataset_name=dataset_name,
                                net_name=net_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    return dataset
