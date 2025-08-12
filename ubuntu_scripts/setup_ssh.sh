#!/bin/bash

# Check if running as root
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 
   exit 1
fi

# Install OpenSSH Server and Git
apt-get update || {
    echo "Failed to update package list. Are you running as root?"
    exit 1
}
apt-get install -y openssh-server git || {
    echo "Failed to install openssh-server and git"
    exit 1
}

# Create required directories
mkdir -p /var/run/sshd
mkdir -p /root/.ssh

# Copy host's SSH keys if available
HOST_SSH_DIR="/host_ssh"
if [ -d "$HOST_SSH_DIR" ]; then
    echo "Copying SSH keys from host..."
    cp -r "$HOST_SSH_DIR"/* /root/.ssh/ 2>/dev/null || echo "No SSH keys found to copy"
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/* 2>/dev/null || true
    echo "SSH keys copied successfully"
else
    echo "Host SSH directory not mounted at $HOST_SSH_DIR - skipping SSH key copy"
fi

# Generate host keys
echo "Generating SSH host keys..."
DEBIAN_FRONTEND=noninteractive ssh-keygen -A
echo "SSH host keys generated successfully"

# Function to read password securely
read_password() {
    local prompt="$1"
    local password=""
    echo -n "$prompt"
    
    # Check if we can use read -s (bash feature)
    if [ -n "$BASH_VERSION" ]; then
        read -s password
    else
        # Fallback for non-bash shells
        stty -echo
        read password
        stty echo
    fi
    echo
    echo "$password"
}

# Check if running in non-interactive mode or if DEFAULT_ROOT_PASSWORD is set
if [ -n "$DEFAULT_ROOT_PASSWORD" ]; then
    echo "Using provided default password..."
    ROOT_PASSWORD="$DEFAULT_ROOT_PASSWORD"
elif [ ! -t 0 ] || [ -n "$DEBIAN_FRONTEND" ]; then
    echo "Non-interactive mode detected. Using default password '123'"
    echo "You can change this by setting DEFAULT_ROOT_PASSWORD environment variable"
    ROOT_PASSWORD="123"
else
    # Prompt for root password with retry logic
    echo "Starting password setup..."
    ROOT_PASSWORD=""
    ROOT_PASSWORD_CONFIRM=""

    for i in {1..3}; do
        ROOT_PASSWORD=$(read_password "Enter new root password: ")
        ROOT_PASSWORD_CONFIRM=$(read_password "Confirm root password: ")
        
        if [ "$ROOT_PASSWORD" = "$ROOT_PASSWORD_CONFIRM" ] && [ -n "$ROOT_PASSWORD" ]; then
            break
        else
            if [ -z "$ROOT_PASSWORD" ]; then
                echo "Password cannot be empty! Try again."
            else
                echo "Passwords do not match! Try again."
            fi
            if [ $i -eq 3 ]; then
                echo "Failed after 3 attempts"
                exit 1
            fi
        fi
    done
fi

# Check if password is still empty
if [ -z "$ROOT_PASSWORD" ]; then
    echo "No password has been supplied."
    exit 1
fi

# Set root password using multiple methods
echo "Setting root password..."

# Method 1: Try chpasswd first
if echo "root:$ROOT_PASSWORD" | chpasswd 2>/dev/null; then
    echo "✓ Root password set successfully using chpasswd"
else
    echo "chpasswd failed, trying passwd instead..."
    
    # Method 2: Try passwd with expect-like behavior
    if command -v expect >/dev/null 2>&1; then
        expect << EOF
spawn passwd root
expect "New password:"
send "$ROOT_PASSWORD\r"
expect "Retype new password:"
send "$ROOT_PASSWORD\r"
expect eof
EOF
        if [ $? -eq 0 ]; then
            echo "✓ Root password set successfully using expect"
        else
            echo "Failed with expect method"
        fi
    else
        # Method 3: Direct passwd approach
        if printf "%s\n%s\n" "$ROOT_PASSWORD" "$ROOT_PASSWORD" | passwd root 2>/dev/null; then
            echo "✓ Root password set successfully using passwd"
        else
            echo "❌ Failed to set root password with all methods"
            echo "You may need to:"
            echo "1. Install expect: apt-get install expect"
            echo "2. Or set password manually: passwd root"
            echo "3. Check if PAM is properly configured"
            # Don't exit here - continue with SSH setup
        fi
    fi
fi

# Configure SSH
echo "Configuring SSH..."
sed -i 's/#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# SSH login fix for containers
sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# Start SSH service
echo "Starting SSH service..."
service ssh start || systemctl start ssh || {
    echo "Failed to start SSH service"
    exit 1
}

echo ""
echo "🎉 Setup complete!"
echo "✓ SSH server installed and configured"
echo "✓ Git installed"
if [ -n "$ROOT_PASSWORD" ]; then
    echo "✓ Root password has been set"
else
    echo "⚠ Root password may need manual configuration"
fi
if [ -d "/host_ssh" ]; then
    echo "✓ SSH keys copied from host"
fi
echo ""
echo "You can now connect using:"
echo "ssh root@<host-ip> -p <mapped-port>"
echo ""
echo "If password authentication fails, you may need to:"
echo "1. Set the password manually: passwd root"
echo "2. Check SSH configuration: cat /etc/ssh/sshd_config"
echo "3. Restart SSH service: service ssh restart"