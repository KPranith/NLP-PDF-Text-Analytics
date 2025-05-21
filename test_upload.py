import http.client
import hashlib
import json


def upload_file():
    # Read file and calculate hash
    with open('testfile', 'rb') as f:
        data = f.read()
    sha256 = hashlib.sha256(data).hexdigest()

    # Create connection
    conn = http.client.HTTPConnection('localhost', 3000)

    # Set up headers
    headers = {
        'Transfer-Encoding': 'chunked',
        'Trailer': 'Content-SHA256'
    }

    # Start request
    conn.putrequest('POST', '/upload')
    for header, value in headers.items():
        conn.putheader(header, value)
    conn.endheaders()

    # Send data in chunks
    chunk_size = 1024 * 1024  # 1MB chunks
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_hex = hex(len(chunk))[2:].encode('ascii')
        conn.send(chunk_hex + b'\r\n' + chunk + b'\r\n')

    # Send trailer
    conn.send(b'0\r\n')
    conn.send(f'Content-SHA256: {sha256}\r\n'.encode('ascii'))
    conn.send(b'\r\n')

    # Get response
    response = conn.getresponse()
    print(json.loads(response.read()))
    conn.close()


if __name__ == '__main__':
    upload_file()