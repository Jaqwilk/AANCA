(() => {
  const story = document.getElementById('learned-story');
  if (!story) return;

  const rows = [...story.querySelectorAll('[data-evidence-step]')];
  const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  const desktopQuery = window.matchMedia('(min-width: 901px)');

  if (reducedQuery.matches || !('IntersectionObserver' in window)) {
    rows.forEach(row => row.classList.add('is-mobile-visible'));
  } else {
    story.classList.add('findings-story--mobile-reveal');
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('is-mobile-visible');
      });
    }, {rootMargin: '0px 0px -12% 0px', threshold: .14});
    rows.forEach(row => observer.observe(row));
  }

  const gsapEngine = window.gsap;
  const scrollEngine = window.ScrollTrigger;
  if (!gsapEngine || !scrollEngine) return;
  gsapEngine.registerPlugin(scrollEngine);

  const media = gsapEngine.matchMedia();
  media.add('(min-width: 901px) and (prefers-reduced-motion: no-preference)', () => {
    const stages = [...story.querySelectorAll('[data-evidence-stage]')];
    const nodes = [...story.querySelectorAll('[data-evidence-node]')];
    const glyphs = [...story.querySelectorAll('[data-summary-glyph]')];
    const atlas = story.querySelector('[data-atlas-preview]');
    const atlasNote = story.querySelector('.findings-story__atlas-note');
    const visual = story.querySelector('.findings-story__visual');
    const progressLine = story.querySelector('.evidence-spine-progress');
    const stageLines = [...story.querySelectorAll('.evidence-stage .evidence-line')];
    if (rows.length !== 7 || stages.length !== 7 || nodes.length !== 7 || !atlas) return;

    story.classList.add('findings-story--enhanced');
    const rowOffset = 300;
    const opacityFor = (index, active) => {
      const distance = index - active;
      if (distance === 0) return 1;
      if (distance === -1) return .34;
      if (distance === 1) return .18;
      return 0;
    };

    gsapEngine.set(rows, {
      yPercent: -50,
      y: index => index * rowOffset,
      autoAlpha: index => opacityFor(index, 0),
    });
    gsapEngine.set(stages, {autoAlpha: index => index === 0 ? 1 : 0});
    gsapEngine.set(nodes, {
      scale: index => index === 0 ? 1.15 : 1,
      fill: index => index === 0 ? '#5e6ad2' : '#010102',
      stroke: index => index === 0 ? '#828fff' : '#62666d',
      opacity: index => index === 0 ? 1 : .42,
    });
    gsapEngine.set(glyphs, {opacity: index => index === 0 ? .72 : .16});
    gsapEngine.set(stageLines, {strokeDashoffset: 1});
    gsapEngine.set([...stages[0].querySelectorAll('.evidence-line')], {strokeDashoffset: 0});
    gsapEngine.set(progressLine, {scaleY: 1 / 7, transformOrigin: '50% 0%'});
    gsapEngine.set(atlas, {autoAlpha: 0, scale: .96, transformOrigin: '50% 50%'});
    gsapEngine.set(atlasNote, {autoAlpha: 0, y: 10});

    const timeline = gsapEngine.timeline({paused: true, defaults: {ease: 'none'}});
    for (let active = 1; active < 7; active += 1) {
      timeline.to(rows, {
        y: index => (index - active) * rowOffset,
        autoAlpha: index => opacityFor(index, active),
        duration: .68,
        ease: 'power2.inOut',
      }, active);
      timeline.to(stages, {
        autoAlpha: index => index === active ? 1 : 0,
        duration: .56,
        ease: 'sine.inOut',
      }, active);
      timeline.to(nodes, {
        scale: index => index === active ? 1.15 : 1,
        fill: index => index === active ? '#5e6ad2' : '#010102',
        stroke: index => index === active ? '#828fff' : '#62666d',
        opacity: index => index === active ? 1 : (index < active ? .56 : .3),
        duration: .5,
        ease: 'power2.inOut',
      }, active);
      timeline.to(glyphs, {
        opacity: index => index === active ? .72 : (index < active ? .32 : .12),
        duration: .5,
      }, active);
      timeline.to(progressLine, {
        scaleY: (active + 1) / 7,
        duration: .62,
        ease: 'power2.inOut',
      }, active);
      const activeLines = [...stages[active].querySelectorAll('.evidence-line')];
      if (activeLines.length) {
        timeline.to(activeLines, {
          strokeDashoffset: 0,
          duration: .58,
          ease: 'power2.inOut',
        }, active + .08);
      }
    }

    timeline.to(rows, {
      y: index => (index - 3) * 30,
      autoAlpha: 0,
      duration: .72,
      ease: 'power2.inOut',
    }, 7);
    timeline.to(stages, {autoAlpha: 0, duration: .42}, 7);
    timeline.to(nodes, {
      scale: 1,
      fill: '#5e6ad2',
      stroke: '#828fff',
      opacity: .78,
      duration: .58,
    }, 7);
    timeline.to(glyphs, {opacity: .58, duration: .58}, 7);
    timeline.to(visual, {scale: .94, duration: .74, ease: 'power2.inOut'}, 7);
    timeline.to(atlas, {
      autoAlpha: 1,
      scale: 1,
      duration: .74,
      ease: 'power2.inOut',
    }, 7.08);
    timeline.to(atlasNote, {
      autoAlpha: 1,
      y: 0,
      duration: .54,
      ease: 'power2.out',
    }, 7.28);

    const trigger = scrollEngine.create({
      trigger: story,
      start: 'top top',
      end: 'bottom bottom',
      animation: timeline,
      scrub: 1.1,
      invalidateOnRefresh: true,
      anticipatePin: 0,
    });

    const refresh = () => scrollEngine.refresh();
    document.fonts?.ready.then(refresh);
    window.addEventListener('load', refresh, {once: true});
    window.addEventListener('resize', refresh, {passive: true});

    return () => {
      window.removeEventListener('resize', refresh);
      trigger.kill();
      timeline.kill();
      story.classList.remove('findings-story--enhanced');
      gsapEngine.set([
        ...rows, ...stages, ...nodes, ...glyphs, atlas, atlasNote, visual, progressLine,
      ], {
        clearProps: 'all',
      });
    };
  });

  const refreshForMediaChange = () => {
    if (desktopQuery.matches && !reducedQuery.matches) scrollEngine.refresh();
  };
  desktopQuery.addEventListener('change', refreshForMediaChange);
  reducedQuery.addEventListener('change', refreshForMediaChange);
})();
