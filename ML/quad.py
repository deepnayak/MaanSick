from dipy.io.image import load_nifti

def depression_quadrant(files):
    q_outs = [0,0,0,0]
    for nii_file in files:
        nii_file = "sample.nii.gz"
        data, affine = load_nifti(nii_file)
        quads = data[:25,:25], data[:25,25:], data[25:,25:], data[25:,:25]
        qsums = [quad.sum() for quad in quads]
        q_outs[qsums.index(max(qsums))] +=1

    quad_dict = {
        0: {"quad": "A","suggestions": ["Logical", "Analytical", "Fact Based", "Quantitative"],"diseases": [],"suggestions": [],"lobes": []},
        1: {"quad": "B","functions": ["Sequential", "Organized", "Detailed", "Planned"],"diseases": [],"suggestions": [],"lobes": []},
        2: {"quad": "C","suggestions": ["Holistic", "Intuitive", "Integrating", "Synthesising"],"diseases": [], "suggestions": [], "lobes": []},
        3: {"quad": "D","suggestions": ["Interpersonal", "Feeling Based", "Kinesthetic", "Emotional"],"diseases": [],"suggestions": [],"lobes": []}
    }

    dep_quad = q_outs.index(max(q_outs))
    output = quad_dict[dep_quad]

    return output



