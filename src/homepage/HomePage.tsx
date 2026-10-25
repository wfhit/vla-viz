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
        </div>
    </div>;
}
