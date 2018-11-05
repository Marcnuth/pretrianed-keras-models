from pkmodels.models import PKModel



def test_main():
    model = PKModel('ssim')
    res = model.predict(files=[
        r'E:\gits\structure-similarity\resources\data\raw\sections\diff\1_0.png',
        r'E:\gits\structure-similarity\resources\data\raw\sections\diff\1_0.png'])

    print(res)