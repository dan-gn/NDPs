
def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False