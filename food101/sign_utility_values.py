import nacl.signing
import nacl.exceptions
import time
import random
import zlib
import struct

class FLClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.signing_key = nacl.signing.SigningKey.generate()
        self.verify_key = self.signing_key.verify_key

    def sign_value(self, value: str):
        start = time.perf_counter()
        signature = self.signing_key.sign(value.encode()).signature
        elapsed = time.perf_counter() - start
        print(f"Client {self.client_id}: Signing took {elapsed:.6f} seconds")
        return signature

    def verify_signature(self, value: str, signature: bytes, verify_key):
        start = time.perf_counter()
        try:
            verify_key.verify(value.encode(), signature)
            elapsed = time.perf_counter() - start
            print(f"Client {self.client_id}: Verification succeeded in {elapsed:.6f} seconds")
            return True
        except nacl.exceptions.BadSignatureError:
            elapsed = time.perf_counter() - start
            print(f"Client {self.client_id}: Verification failed in {elapsed:.6f} seconds")
            return False

class FLServer:
    def __init__(self, chunk_size=100):
        self.values = {}
        self.signatures = {}
        self.chunk_size = chunk_size
        self.compressed_chunks = {}

    def collect_data(self, clients):
        for client in clients:
            numeric_value = random.randint(1000, 9999)
            signature = client.sign_value(str(numeric_value))
            self.values[client.client_id] = numeric_value
            self.signatures[client.client_id] = signature

    def broadcast_data(self):
        value_str = b''.join(struct.pack('!H', val) for val in self.values.values())
        cids = list(self.signatures.keys())
        for i in range(0, len(cids), self.chunk_size):
            chunk_cids = cids[i:i+self.chunk_size]
            chunk_data = b''.join(self.signatures[cid] for cid in chunk_cids)
            compressed_chunk = zlib.compress(chunk_data, level=9)
            self.compressed_chunks[i // self.chunk_size] = compressed_chunk

        total_compressed_size = sum(len(chunk) for chunk in self.compressed_chunks.values())
        print(f"Concatenated numeric values size: {len(value_str)} bytes")
        print(f"Total compressed signatures size: {total_compressed_size} bytes")

        return value_str, self.compressed_chunks

# Utility function to verify signature given chunk position
def verify_signature_from_chunk(client_index, chunk_size, compressed_chunks, value_str, public_key):
    start = time.perf_counter()
    signature_size = 64
    chunk_index = client_index // chunk_size
    offset = (client_index % chunk_size) * signature_size

    decompressed_chunk = zlib.decompress(compressed_chunks[chunk_index])
    signature = decompressed_chunk[offset:offset + signature_size]
    numeric_value = struct.unpack_from('!H', value_str, client_index * 2)[0]

    try:
        public_key.verify(str(numeric_value).encode(), signature)
        elapsed = time.perf_counter() - start
        print(f"Verification succeeded in {elapsed:.6f} seconds\n")
        print(f"Verification succeeded for client at index {client_index}.")
        return True
    except nacl.exceptions.BadSignatureError:
        print(f"Verification failed for client at index {client_index}.")
        return False

if __name__ == "__main__":
    num_clients = 10000
    clients = [FLClient(f"client_{i}") for i in range(num_clients)]
    server = FLServer(chunk_size=100)

    server.collect_data(clients)

    value_str, compressed_chunks = server.broadcast_data()

    random_client = random.choice(clients)
    client_index = int(random_client.client_id.split('_')[1])

    verify_signature_from_chunk(client_index, server.chunk_size, compressed_chunks, value_str, random_client.verify_key)
