'use client';

import React from 'react';
import { faCircleQuestion } from '@fortawesome/free-regular-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { createContext, useContext, useEffect } from 'react';
import { assignImm } from '@/src/utils/data';
import { KeyboardOrder, useGlobalKeyboard } from '@/src/utils/keyboard';
import { useLocalStorageState } from '@/src/utils/localstorage';
import { ModalWindow } from '@/src/utils/Portal';
import s from './WelcomePopup.module.scss';
import { TocDiagram } from './components/TocDiagram';
import { Subscriptions, useSubscriptions } from '../utils/hooks';

interface IWelcomePopupLS {
    visible: boolean;
}

function hydrateWelcomePopupLS(a?: Partial<IWelcomePopupLS>) {
    return {
        visible: a?.visible ?? true,
    };
}

export const WelcomePopup: React.FC<{}> = () => {
    let ctx = useContext(WelcomeContext);
    useSubscriptions(ctx.subscriptions);
    let [welcomeState, setWelcomeState] = useLocalStorageState('welcome-popup', hydrateWelcomePopupLS);

    useGlobalKeyboard(KeyboardOrder.Modal, ev => {

        if (ev.key === 'Escape') {
            hide();
        }

        ev.stopPropagation();
    });

    useEffect(() => {
        if (ctx.forceVisible) {
            ctx.forceVisible = false;
            setWelcomeState(a => assignImm(a, { visible: true }));
        }
    }, [ctx, setWelcomeState, ctx.forceVisible]);

    function hide() {
        setWelcomeState(a => assignImm(a, { visible: false }));
    }

    if (!welcomeState.visible) {
        return null;
    }

    return <ModalWindow className={s.modalWindow} backdropClassName={s.modalWindowBackdrop} onBackdropClick={hide}>
        <div className={s.header}>
            <div className={s.title}>欢迎！</div>
        </div>
        <div className={s.body}>
            {/* <div className={s.image}>
                <Image src={IntroImage} alt={"LLM diagram"} />
            </div> */}
            <div style={{ width: 600, flex: '0 0 auto' }}>
                <TocDiagram activePhase={null} onEnterPhase={hide} />
            </div>
            <div className={s.text}>
                <p>这是 k005b VLA 网络的交互式 3D 可视化：相机 patch + 语言 + 状态 → Qwen3-VL 主干 → knack 分类头 / done 头 / 动作专家。</p>
                <p>3D 场景中的小模型只是示意（真实 k005b 为 36 层、宽度 768）。</p>
                <p>你可以自由交互，也可以跟随讲解逐步查看每个环节。</p>
            </div>
        </div>
        <div className={s.footer}>
            <button className={s.button} onClick={hide}>开始</button>
        </div>
    </ModalWindow>;
};

class WelcomeManager {
    subscriptions = new Subscriptions();
    forceVisible = false;
    showWelcomeDialog() {
        this.forceVisible = true;
        this.subscriptions.notify();
    }
}

let WelcomeContext = createContext(new WelcomeManager());

export const InfoButton: React.FC<{}> = () => {
    let ctx = useContext(WelcomeContext);

    return <div onClick={() => ctx.showWelcomeDialog()} className={s.infoBtn}>
        <FontAwesomeIcon icon={faCircleQuestion} />
    </div>;
};
