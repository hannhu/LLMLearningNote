# Ubuntu Kernel Version Switching

**Problem**: Ubuntu kernel updates can cause boot black screen issues.  
**Solution**: Switch to a previous working kernel version.

## 1. Check Current Kernel Version
```bash
uname -r
```
Output:
```bash
5.15.0-67-generic
```

## 2. Find Available Kernels
```bash
grep gnulinux /boot/grub/grub.cfg
```
Output:
```bash
if [ x"${feature_menuentry_id}" = xy ]; then
  menuentry_id_option="--id"
  menuentry_id_option=""
export menuentry_id_option
menuentry 'Ubuntu' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-simple-ee28ccb6-5568-40a0-b312-133865a7fac1' {
submenu 'Advanced options for Ubuntu' $menuentry_id_option 'gnulinux-advanced-ee28ccb6-5568-40a0-b312-133865a7fac1' {
        menuentry 'Ubuntu, with Linux 5.15.0-88-generic' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-5.15.0-88-generic-advanced-ee28ccb6-5568-40a0-b312-133865a7fac1' {
        menuentry 'Ubuntu, with Linux 5.15.0-88-generic (recovery mode)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-5.15.0-88-generic-recovery-ee28ccb6-5568-40a0-b312-133865a7fac1' {
        menuentry 'Ubuntu, with Linux 5.15.0-67-generic' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-5.15.0-67-generic-advanced-ee28ccb6-5568-40a0-b312-133865a7fac1' {
        menuentry 'Ubuntu, with Linux 5.15.0-67-generic (recovery mode)' --class ubuntu --class gnu-linux --class gnu --class os $menuentry_id_option 'gnulinux-5.15.0-67-generic-recovery-ee28ccb6-5568-40a0-b312-133865a7fac1' {
menuentry 'UEFI Firmware Settings' $menuentry_id_option 'uefi-firmware' {
```

Record the submenu ID:
```bash
gnulinux-advanced-ee28ccb6-5568-40a0-b312-133865a7fac1
```

Select target kernel (Ubuntu, with Linux 5.15.0-67-generic) with ID:
```bash
gnulinux-5.15.0-67-generic-advanced-ee28ccb6-5568-40a0-b312-133865a7fac1
```

## 3. Modify GRUB Configuration
```bash
sudo vim /etc/default/grub
```

Change `GRUB_DEFAULT=0` to:
```bash
GRUB_DEFAULT="gnulinux-advanced-ee28ccb6-5568-40a0-b312-133865a7fac1>gnulinux-5.15.0-67-generic-advanced-ee28ccb6-5568-40a0-b312-133865a7fac1"
```

Alternative method (if above fails):
```bash
GRUB_DEFAULT="1>2"
```
This uses the 3rd kernel in the submenu (Ubuntu, with Linux 5.15.0-67-generic) as default.

## 4. Update GRUB
```bash
sudo update-grub
```

## 5. Reboot
```bash
sudo reboot
```

---
**Source**: [Original CSDN Article](https://blog.csdn.net/m0_46249060/article/details/134291880)