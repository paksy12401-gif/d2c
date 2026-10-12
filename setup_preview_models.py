"""One-time public OCR model download. Never accepts or opens drawing files."""
if __name__ == '__main__':
    import easyocr
    from ocr_local import MODEL_DIR
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    easyocr.Reader(['ko','en'],gpu=False,model_storage_directory=str(MODEL_DIR),
                   user_network_directory=str(MODEL_DIR/'user_network'),
                   download_enabled=True,verbose=False)
    print('Korean/English local OCR models ready.')
