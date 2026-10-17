"""Command boundaries for offline research runs. No data access at import."""
import argparse
import json
import sys
from .inputs import REPO


def parser():
    p=argparse.ArgumentParser(prog='cbam-power', description='Offline CBAM electricity research; demo is artificial.')
    sub=p.add_subparsers(dest='command', required=True)
    for name in ['doctor','validate-inputs','build-weather','build-panel','build-synthetic',
                 'estimate','verify-baseline','report','demo','reproduce']:
        q=sub.add_parser(name)
        q.add_argument('--mode',choices=['demo','authorized'],default='demo' if name in ['doctor','demo'] else 'authorized')
        q.add_argument('--input-root'); q.add_argument('--output-root'); q.add_argument('--run-id')
        q.add_argument('--project-root',default=str(REPO))
        if name in ['estimate','verify-baseline']:
            q.add_argument('--spec',choices=['main','polynomial']+(['all'] if name=='verify-baseline' else []),required=True)
        if name=='report': q.add_argument('--audience',choices=['local','public'],default='local')
    for name in ['release-check', 'export-public']:
        q=sub.add_parser(name)
        q.add_argument('--registry', default=str(REPO/'manifests/release_rights.toml'))
        q.add_argument('--context', choices=['public_repository','research_publication','commercial_reuse'], default='public_repository')
        q.add_argument('--conditions', help='Reviewed JSON map of condition IDs to evidence IDs')
        if name=='release-check':
            q.add_argument('--artifact', required=True)
        else:
            q.add_argument('--source-root', required=True)
            q.add_argument('--staging', required=True)
            q.add_argument('--selection', required=True, help='Explicit JSON map of relative file paths to registry node IDs')
            q.add_argument('--protected-root', action='append', default=[])
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    from .pipeline import Run, doctor, demo, reproduce, run_stage
    try:
        if args.command in {'release-check','export-public'}:
            from pathlib import Path
            import os
            from .release import RightsRegistry, stage_public
            registry=RightsRegistry(args.registry)
            conditions=json.loads(Path(args.conditions).read_text()) if args.conditions else None
            if args.command=='release-check':
                result=registry.authorize(args.artifact, context=args.context,
                                          satisfied_conditions=conditions)
            else:
                selection=json.loads(Path(args.selection).read_text())
                protected=[REPO, *args.protected_root]
                if os.environ.get('CBAM_INPUT_ROOT'): protected.append(os.environ['CBAM_INPUT_ROOT'])
                result=stage_public(registry,args.source_root,args.staging,whitelist=selection,
                                    context=args.context,satisfied_conditions=conditions,
                                    protected_roots=protected)
            # Never disclose rejected file contents or local source paths.
            print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
        if args.command=='doctor':
            result=doctor(mode=args.mode,project_root=args.project_root,input_root=args.input_root,output_root=args.output_root)
        else:
            if args.command not in {'demo','reproduce','validate-inputs'} and not args.run_id:
                raise ValueError('EXPLICIT_RUN_ID_REQUIRED_FOR_EXISTING_STAGE')
            mode='demo' if args.command=='demo' else args.mode
            if args.command=='reproduce' and mode!='authorized': raise ValueError('AUTHORIZED_MODE_REQUIRED')
            run=Run(mode=mode,run_id=args.run_id,input_root=args.input_root,
                    output_root=args.output_root,project_root=args.project_root)
            if args.command=='demo': result=demo(run)
            elif args.command=='reproduce': result=reproduce(run)
            elif args.command=='estimate': result=run_stage(run,'estimate-'+args.spec)
            elif args.command=='verify-baseline':
                specs=['main','polynomial'] if args.spec=='all' else [args.spec]
                result={s:run_stage(run,'verify-'+s) for s in specs}
            elif args.command=='report': result=run_stage(run,'report',audience=args.audience)
            else: result=run_stage(run,args.command)
            result=dict(run_id=run.receipt['run_id'],mode=mode,
                        completed_stages=run.receipt.get('completed_stages', []),
                        status='PASS', label=run.receipt['label'],
                        evidence='Detailed stage identities and artifact hashes are in the private run.json')
        print(json.dumps(result,ensure_ascii=False,default=str,indent=2)); return 0
    except (ValueError,RuntimeError,OSError,AssertionError,KeyError) as exc:
        # Full local exception details belong in the private run receipt. CLI
        # errors give a nonzero status; never switch to demo or fetch inputs.
        if args.command in {'release-check','export-public'}:
            from .release import public_error_code
            print(public_error_code(exc),file=sys.stderr)
        else:
            print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
