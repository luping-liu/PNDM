import sys
import copy
import torch as th


def choose_method(name):
    if name == 'DDIM':
        return gen_order_1
    elif name == 'S-PNDM':
        return gen_order_2
    elif name == 'F-PNDM':
        return gen_order_4
    else:
        return None


def gen_order_4(img, t, t_next, model, alphas_cump, ets):
    t_list = [t, (t+t_next)/2, t_next]
    if len(ets) > 2:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = runge_kutta(img, t_list, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def runge_kutta(x, t_list, model, alphas_cump, ets):
    e_1 = model(x, t_list[0])
    ets.append(e_1)
    x_2 = transfer(x, t_list[0], t_list[1], e_1, alphas_cump)

    e_2 = model(x_2, t_list[1])
    x_3 = transfer(x, t_list[0], t_list[1], e_2, alphas_cump)

    e_3 = model(x_3, t_list[1])
    x_4 = transfer(x, t_list[0], t_list[2], e_3, alphas_cump)

    e_4 = model(x_4, t_list[2])
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et


def gen_order_2(img, t, t_next, model, alphas_cump, ets):
    if len(ets) > 0:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = 0.5 * (3 * ets[-1] - ets[-2])
    else:
        noise = improved_eular(img, t, t_next, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def improved_eular(x, t, t_next, model, alphas_cump, ets):
    e_1 = model(x, t)
    ets.append(e_1)
    x_2 = transfer(x, t, t_next, e_1, alphas_cump)

    e_2 = model(x_2, t_next)
    et = (e_1 + e_2) / 2
    # x_next = transfer(x, t, t_next, et, alphas_cump)

    return et


def gen_order_1(img, t, t_next, model, alphas_cump, ets):
    noise = model(img, t)
    ets.append(noise)
    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def transfer(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

    x_next = x + x_delta
    return x_next