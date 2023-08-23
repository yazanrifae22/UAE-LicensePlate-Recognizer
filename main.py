import cv2
import logging
import argparse
from multiprocessing import Process
from plate_processing import process_stream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(description='License Plate Recognition')
    parser.add_argument('stream_url', type=str, help='URL of the video stream')
    parser.add_argument('camera_mac_address', type=str, help='Camera MAC address')
    args = parser.parse_args()

    logging.info('Starting License Plate Recognition...')
    
    try:
        process = Process(target=process_stream, args=(args.stream_url, args.camera_mac_address))
        process.start()
        while True:
            try:
                user_input = input("Enter a command (q to quit): ").strip().lower()
                
                if user_input == 'q':
                    logging.info('Exiting License Plate Recognition...')
                    process.terminate()
                    break
                else:
                    logging.warning('Invalid command. Please enter "q" to quit.')
            
            except KeyboardInterrupt:
                logging.info('Exiting License Plate Recognition...')
                process.terminate()
                break
    except KeyboardInterrupt:
        logging.info('Exiting License Plate Recognition...')
        process.terminate()

if __name__ == "__main__":
    main()
