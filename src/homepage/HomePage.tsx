'use client';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import React from 'react';
import s from './HomePage.module.scss';

export const HomePage: React.FC = () => {
    let router = useRouter();

    return <div className={s.homePage}>
        <div className={s.headerSection}>
            <div className={s.title}>VLA 可视化</div>
            <div className={s.subtitle}>真实权重 · 真实激活 · 逐层 3D 走查</div>
        </div>

        <div className={s.projectsSection}>
            <div className={s.projectCard} onClick={() => router.push('/vla')}>
                <div className={s.cardImageWrapper}>
                    <div className={s.cardImage}>
                        <img src="/images/vla-viz-screenshot.jpg" alt="k005b VLA 场景截图" />
                    </div>
                </div>
                <div className={s.cardContent}>
                    <div className={s.cardTitle}>
                        <Link href="/vla">k005b VLA</Link>
                    </div>
                    <div>
                        Qwen3-VL 主干 28 层 + knack/done 头 + DiT 动作专家（联合注意力、AdaRMSNorm 时间条件），
                        10 步 flow matching 解码出 16 × 23 动作块。全部来自真实 k005b 权重和真实激活。
                    </div>
                </div>
            </div>
            <div className={s.projectCard} onClick={() => router.push('/groot')}>
                <div className={s.cardImageWrapper}>
                    <div className={s.cardImage}>
                        <img src="/images/groot-viz-screenshot.jpg" alt="GR00T N1.7 场景截图" />
                    </div>
                </div>
                <div className={s.cardContent}>
                    <div className={s.cardTitle}>
                        <Link href="/groot">GR00T N1.7</Link>
                    </div>
                    <div>
                        System 2：Cosmos-Reason2 视觉语言主干（16 层）；System 1：32 块 DiT 动作头（文本/图像交叉注意力与自注意力交替、AdaLN 时间条件），
                        4 步 Euler 解码出 40 × 132 动作块。真实 GR00T-N1.7-3B 权重，BEHAVIOR 示范帧上的真实激活。
                    </div>
                </div>
            </div>
            <div className={s.projectCard} onClick={() => router.push('/pi0')}>
                <div className={s.cardImageWrapper}>
                    <div className={s.cardImage}>
                        <img src="/images/pi0-viz-screenshot.jpg" alt="π0.5 场景截图" />
                    </div>
                </div>
                <div className={s.cardContent}>
                    <div className={s.cardTitle}>
                        <Link href="/pi0">π0.5</Link>
                    </div>
                    <div>
                        PaliGemma 前缀（SigLIP + Gemma 2B，18 层）+ 300M 动作专家，逐层联合注意力、adaRMSNorm 时间条件，
                        状态离散化进提示词，10 步 Euler 解码出 32 × 32 动作块（R1 Pro 23 维）。openpi-comet BEHAVIOR 微调检查点的真实权重与激活。
                    </div>
                </div>
            </div>
        </div>
    </div>;
}
