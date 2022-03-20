losses = ['l1', 'poisson', 'psnr', 'ssim', 'ms_ssim']

print_l = '['
for loss in losses:
    print_l += "['" + str(loss) + "']" + ", "

for i in range(len(losses)):
    loss_1 = losses[i]
    for j in range(i+1, len(losses)):
        loss_2 = losses[j]
        print_l += f"['{loss_1}', '{loss_2}'], "

print_l = print_l[:-2]
print_l += ']'

print(print_l)