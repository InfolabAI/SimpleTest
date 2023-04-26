try:
    raise Exception()
except:
    import traceback
    print(traceback.format_exc())
