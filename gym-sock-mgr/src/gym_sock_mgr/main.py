from gym_sock_mgr.gymsockmgr import GymSocketMgr


if __name__ == '__main__':
    print("Starting ...\n")
    
    while True:
        gym_socket_mgr = GymSocketMgr()
        gym_socket_mgr.safely_run()
        print("\nOperation Complete!")
