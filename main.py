def main():
    from reverse_engine import reverse_engineer 
    
    result = reverse_engineer('./Challenging Car Restorations Car SOS.mp4', output_dir='./output') 
    print(result)


if __name__ == "__main__":
    main()
