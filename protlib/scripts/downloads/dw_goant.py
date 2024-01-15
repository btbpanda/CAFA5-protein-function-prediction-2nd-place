import argparse
import os
import subprocess
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str)

if __name__ == '__main__':

    args = parser.parse_args()
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    output = os.path.join(config['base_path'], config['temporal_path'])
    os.makedirs(output, exist_ok=True)

    procs = []

    # for i in [199, 211, 213, 214, 215]:
    # 216 is actual data on the end of competition
    # 214 needed to reproduce cafa-terms-diff dataset
    for i in [214, 216]:  # on the moment of competition 216 was actual, but now moved to history
        url = f'http://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/goa_uniprot_all.gaf.{i}.gz'
        path = os.path.join(output, f'goa_uniprot_all.gaf.{i}.gz')

        if not os.path.exists(path):
            p = subprocess.Popen(
                f'wget {url} -O {path}'.split()
            )
            procs.append(p)

    # url = f'http://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz'
    # path = os.path.join(output, f'goa_uniprot_all.gaf.216.gz')
    #
    # if not os.path.exists(path):
    #     p = subprocess.Popen(
    #         f'wget {url} -O {path}'.split()
    #     )
    #     procs.append(p)

    [x.wait() for x in procs]
