from pymodbus.client import ModbusTcpClient
class ModbusTCP:
    def __init__(self, host='192.168.1.100', port=502):
        self.client = ModbusTcpClient(host=host, port=port)

    def connect(self):
        self.client.connect()
        return self.client.connected

    def read_status_output(self, address): #function code 01
        response = self.client.read_coils(address, 1)
        if response.isError():
            raise Exception(f"Error reading digital coil at address {address}: {response}")
        return response.bits[0]
    
    def digital_write(self, address, value): #function code 05
        response = self.client.write_coil(address, value)
        if response.isError():
            raise Exception(f"Error writing digital coil at address {address}: {response}")
        return response
    
    def multiple_digital_write(self, address, values = [0, 0, 0, 0]): #function code 15
        response = self.client.write_coils(address, values)
        if response.isError():
            raise Exception(f"Error writing multiple digital coils at address {address}: {response}")
        return response
    
    def digital_input(self, address, count=1): #function code 02
        response = self.client.read_discrete_inputs(address=address, count=count)
        print(response)
        if response.isError():
            raise Exception(f"Error reading digital input at address {address}: {response}")
        return response.bits[0]

    def analog_read(self, address, count=1): #function code 04
        response = self.client.read_input_registers(address=address, count=count)
        print(response)
        if response.isError():
            raise Exception(f"Error reading analog input at address {address}: {response}")
        return response.registers[0]

    def read_holding_registers(self, address, count=1, slave_id=1): #function code 03
        response = self.client.read_holding_registers(address=address, count=count, unit=slave_id)
        print(response)
        if response.isError():
            raise Exception(f"Error reading holding registers at address {address}: {response}")
        return response.registers
    
    def write_holding_registers(self, address, values): #function code 06
        response = self.client.write_registers(address, values)
        if response.isError():
            raise Exception(f"Error writing holding registers at address {address}: {response}")
        return response
    
    def multiple_write_holding_registers(self, address, values=[0,0]): #function code 16
        response = self.client.write_registers(address, values)
        if response.isError():
            raise Exception(f"Error writing multiple holding registers at address {address}: {response}")
        return response
    
    def disconnect(self):
        self.client.close()
        return self.client.connected
